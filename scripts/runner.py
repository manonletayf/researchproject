"""Task runner - executes multi-agent tasks and captures traces for metric evaluation."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agentcollabbench.harness.injector import (
    build_cpr_nonseed_system_prompt_patch,
    build_cpr_system_prompt_patch,
    build_idr_system_prompt_patch,
    build_injection_prompt,
)
from agentcollabbench.harness.provider import (
    RQ2_TYPED_RESPONSE_SCHEMA,
    SCHEMA_MODE_SUPPORTED_PROVIDERS,
    LLMProvider,
    Message,
)
from agentcollabbench.metrics.rtd.topology import DAGTopology

logger = logging.getLogger(__name__)


@dataclass
class Handoff:
    """A single inter-agent message."""

    sender: str
    receiver: str
    content: str
    timestamp: float = 0.0


@dataclass
class RunResult:
    """Complete result of running a task through agents."""

    task_id: str
    scenario: dict[str, Any]
    output: dict[str, Any]
    trace: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "scenario": self.scenario,
            "output": self.output,
            "trace": self.trace,
        }


class TaskRunner:
    """Runs a multi-agent task through LLM agents and captures the trace.

    Takes a task JSON from ``tasks/``, instantiates agents per the topology,
    runs the conversation following the DAG edges, and produces the
    (scenario, output, trace) tuple that metrics consume.
    """

    def __init__(
        self,
        provider: LLMProvider,
        metric: str = "rtd",
        verbose: bool = False,
        schema_mode: str = "free_text",
        response_schema: dict[str, Any] | None = None,
    ):
        """
        Args:
            provider: The LLM provider to use for all agents.
            metric: Which metric injection to apply (default "rtd").
            verbose: If True, log agent messages to stdout.
            schema_mode: "free_text" (default, unchanged behavior) or "typed".
                In "typed" mode, every agent turn is requested as structured
                JSON conforming to `response_schema` via the provider's
                native structured-output support (see provider.py). Use
                "free_text" and "typed" as the two levels of RQ2's message
                schema factor.
            response_schema: JSON-schema dict (with "name"/"schema" keys)
                used when schema_mode == "typed". Defaults to
                RQ2_TYPED_RESPONSE_SCHEMA, whose "findings" field is the
                designated free-text carrier for RTD tracers/anchors.
        """
        if schema_mode not in ("free_text", "typed"):
            raise ValueError(
                f"schema_mode must be 'free_text' or 'typed', got {schema_mode!r}"
            )
        self.provider = provider
        self.metric = metric
        self.verbose = verbose
        self.schema_mode = schema_mode
        self.response_schema = response_schema or RQ2_TYPED_RESPONSE_SCHEMA

        if self.schema_mode == "typed" and provider.provider_name() not in SCHEMA_MODE_SUPPORTED_PROVIDERS:
            logger.warning(
                "schema_mode='typed' requested with provider '%s', which is not "
                "known to enforce response_schema natively. Calls will likely "
                "silently return free text, contaminating the 'typed' condition "
                "of RQ2. Supported providers: %s",
                provider.provider_name(),
                sorted(SCHEMA_MODE_SUPPORTED_PROVIDERS),
            )

    def run_task(self, task: dict[str, Any]) -> RunResult:
        """Execute a single task and return the result.

        Args:
            task: The full task JSON dict loaded from a dataset file.

        Returns:
            RunResult containing scenario, output, and trace.
        """
        task_id = task.get("task_id", "unknown")
        topology_data = task.get("topology", {})
        topo = DAGTopology(topology_data)

        agents_config = {a["agent_id"]: a for a in topology_data.get("agents", [])}
        speaking_order = topology_data.get("speaking_order", list(agents_config.keys()))

        agent_outputs: dict[str, str] = {}
        handoffs: list[Handoff] = []
        intermediate_steps: list[dict[str, Any]] = []
        api_calls: list[dict[str, Any]] = []

        if self.metric == "clc":
            self._run_clc_task_b_pair(
                task,
                topo,
                agents_config,
                speaking_order,
                agent_outputs,
                handoffs,
                intermediate_steps,
                api_calls,
            )
        else:
            injection_prompt = build_injection_prompt(task, self.metric)
            self._run_speaking_order(
                task,
                topo,
                agents_config,
                speaking_order,
                injection_prompt,
                agent_outputs,
                handoffs,
                intermediate_steps,
                api_calls,
            )

        # Build the final result
        leaf_agents = topo.leaf_agents()
        final_outputs = [agent_outputs[a] for a in leaf_agents if a in agent_outputs]
        if not final_outputs and intermediate_steps:
            final_outputs = [intermediate_steps[-1]["content"]]
        final_result = "\n\n---\n\n".join(final_outputs) if final_outputs else ""

        trace_overrides = task.get("_runner_trace_overrides", {})
        scenario = {k: v for k, v in task.items() if k != "_runner_trace_overrides"}
        output = {
            "result": final_result,
            "intermediate_steps": intermediate_steps,
        }
        trace = {
            "turns": intermediate_steps,
            "conversation": [
                {"agent_id": s["agent_id"], "role": s.get("role", ""), "content": s["content"]}
                for s in intermediate_steps
            ],
            "handoffs": [
                {
                    "sender": h.sender,
                    "receiver": h.receiver,
                    "content": h.content,
                    "timestamp": h.timestamp,
                }
                for h in handoffs
            ],
            "api_calls": api_calls,
            "timestamps": {
                "start": handoffs[0].timestamp if handoffs else 0,
                "end": handoffs[-1].timestamp if handoffs else 0,
                # NOTE: _run_speaking_order executes agents strictly in
                # sequence regardless of topology (fully_connected /
                # branching_tree / custom_graph all go through the same
                # `for agent_id in speaking_order` loop). This sum is
                # therefore total sequential LLM compute time, not a measure
                # of concurrency benefit — a decentralised topology will not
                # show lower total_llm_latency_seconds than a hierarchical
                # one purely from parallelism, since none occurs here. Treat
                # it as "compute cost", and treat end-start above as wall
                # clock span, when writing up coordination latency.
                "total_llm_latency_seconds": round(
                    sum(c.get("latency_seconds", 0.0) for c in api_calls), 4
                ),
            },
            "errors": [],
        }
        trace.update(trace_overrides)

        return RunResult(
            task_id=task_id,
            scenario=scenario,
            output=output,
            trace=trace,
        )

    def _run_clc_task_b_pair(
        self,
        task: dict[str, Any],
        topo: DAGTopology,
        agents_config: dict[str, dict[str, Any]],
        speaking_order: list[str],
        agent_outputs: dict[str, str],
        handoffs: list[Handoff],
        intermediate_steps: list[dict[str, Any]],
        api_calls: list[dict[str, Any]],
    ) -> None:
        """Run Task A then Task B in one session and retain Task B output."""
        clc = task.get("injections", {}).get("clc", {})
        task_a_prompt = build_injection_prompt(task, "clc")
        task_b_prompt = clc.get("task_b_description") or clc.get("vacuum_probe", {}).get(
            "task_b_description", ""
        )
        if not task_b_prompt:
            task_b_prompt = "Now move to an unrelated task. Discuss general improvements only."

        self._run_speaking_order(
            task,
            topo,
            agents_config,
            speaking_order,
            task_a_prompt,
            agent_outputs,
            handoffs,
            intermediate_steps,
            api_calls,
            phase="task_a",
            expand_to_expected=False,
        )
        before_task_b = len(intermediate_steps)
        self._run_speaking_order(
            task,
            topo,
            agents_config,
            speaking_order,
            task_b_prompt,
            agent_outputs,
            handoffs,
            intermediate_steps,
            api_calls,
            phase="task_b",
            expand_to_expected=False,
        )
        task_b_text = "\n\n".join(step["content"] for step in intermediate_steps[before_task_b:])
        task.setdefault("_runner_trace_overrides", {})["task_b_output"] = task_b_text

    def _run_speaking_order(
        self,
        task: dict[str, Any],
        topo: DAGTopology,
        agents_config: dict[str, dict[str, Any]],
        speaking_order: list[str],
        root_prompt: str,
        agent_outputs: dict[str, str],
        handoffs: list[Handoff],
        intermediate_steps: list[dict[str, Any]],
        api_calls: list[dict[str, Any]],
        phase: str | None = None,
        expand_to_expected: bool = True,
    ) -> None:
        pending_pressures = self._idr_peer_pressure_events(task) if self.metric == "idr" else []
        inserted_pressure_indices: set[int] = set()
        target_turn_counts: dict[str, int] = {}

        for agent_id in self._effective_speaking_order(task, speaking_order, expand_to_expected):
            self._append_due_idr_pressure_events(
                task=task,
                pending_pressures=pending_pressures,
                inserted_pressure_indices=inserted_pressure_indices,
                intermediate_steps=intermediate_steps,
                phase=phase,
            )

            config = agents_config.get(agent_id)
            if not config:
                logger.warning("Agent %s in speaking_order but not in agents list", agent_id)
                continue

            system_prompt = self._system_prompt_for_metric(config, agent_id, task)
            parents = topo.parents_of(agent_id)
            parent_context = self._assemble_parent_context(parents, agent_outputs)
            user_prompt = parent_context or root_prompt
            trigger_type = self._idr_trigger_type_for_turn(
                task,
                agent_id,
                target_turn_counts.get(agent_id, 0),
            )
            user_prompt = self._augment_user_prompt_for_metric(task, user_prompt, trigger_type)

            messages = [
                Message(role="system", content=system_prompt),
                Message(role="user", content=user_prompt),
            ]

            if self.verbose:
                logger.info("[%s (%s)] Calling LLM...", agent_id, config.get("role", ""))

            call_kwargs: dict[str, Any] = {}
            if self.schema_mode == "typed":
                call_kwargs["response_schema"] = self.response_schema

            call_start = time.perf_counter()
            response = self.provider.chat(messages, **call_kwargs)
            llm_call_latency_seconds = time.perf_counter() - call_start

            agent_output = response.content
            agent_outputs[agent_id] = agent_output
            timestamp = time.time()

            if self.verbose:
                role = config.get("role", agent_id)
                logger.info("[%s (%s)]:\n%s...", agent_id, role, agent_output[:300])

            step = {
                "agent_id": agent_id,
                "role": config.get("role", ""),
                "content": agent_output,
            }
            if phase:
                step["phase"] = phase
            if trigger_type != "normal":
                step["trigger_type"] = trigger_type
            intermediate_steps.append(step)
            target_turn_counts[agent_id] = target_turn_counts.get(agent_id, 0) + 1

            api_calls.append(
                {
                    "agent_id": agent_id,
                    "provider": self.provider.provider_name(),
                    "model": response.model,
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                    "latency_seconds": round(llm_call_latency_seconds, 4),
                    "schema_mode": self.schema_mode,
                }
            )

            children = topo.children_of(agent_id)
            for child_id in children:
                handoffs.append(
                    Handoff(
                        sender=agent_id,
                        receiver=child_id,
                        content=agent_output,
                        timestamp=timestamp,
                    )
                )

            if not children and parents:
                handoffs.append(
                    Handoff(
                        sender=agent_id,
                        receiver="__output__",
                        content=agent_output,
                        timestamp=timestamp,
                    )
                )

        self._append_due_idr_pressure_events(
            task=task,
            pending_pressures=pending_pressures,
            inserted_pressure_indices=inserted_pressure_indices,
            intermediate_steps=intermediate_steps,
            phase=phase,
            flush=True,
        )

    def _effective_speaking_order(
        self,
        task: dict[str, Any],
        speaking_order: list[str],
        expand_to_expected: bool,
    ) -> list[str]:
        """Repeat the speaking order until the metric's expected turn budget is met."""
        if not expand_to_expected or not speaking_order:
            return speaking_order

        injection = task.get("injections", {}).get(self.metric, {})
        expected_turns = int(
            injection.get("expected_turns") or task.get("expected_turns") or len(speaking_order)
        )
        if expected_turns <= len(speaking_order):
            return speaking_order

        return [speaking_order[i % len(speaking_order)] for i in range(expected_turns)]

    def _idr_peer_pressure_events(self, task: dict[str, Any]) -> list[dict[str, Any]]:
        idr = task.get("injections", {}).get("idr", {})
        events = idr.get("peer_pressure_turns", [])
        if not isinstance(events, list):
            return []
        return sorted(
            [e for e in events if isinstance(e, dict) and e.get("message")],
            key=lambda e: int(e.get("turn_index", 0)),
        )

    def _append_due_idr_pressure_events(
        self,
        task: dict[str, Any],
        pending_pressures: list[dict[str, Any]],
        inserted_pressure_indices: set[int],
        intermediate_steps: list[dict[str, Any]],
        phase: str | None,
        flush: bool = False,
    ) -> None:
        """Expose scheduled IDR peer-pressure stimuli as trace turns.

        Peer-pressure turns are experiment stimuli, not model calls, so they are
        recorded as synthetic turns and excluded from API-call accounting.
        """
        if self.metric != "idr":
            return

        for index, event in enumerate(pending_pressures):
            if index in inserted_pressure_indices:
                continue
            scheduled_index = int(event.get("turn_index", 0))
            if not flush and scheduled_index > len(intermediate_steps):
                continue

            step = {
                "agent_id": event.get("speaker", "peer_pressure"),
                "role": "peer_pressure",
                "content": str(event["message"]),
                "synthetic": True,
                "phase": phase or "peer_pressure",
                "target_agent": task.get("injections", {}).get("idr", {}).get("target_agent", ""),
            }
            intermediate_steps.append(step)
            inserted_pressure_indices.add(index)

    def _idr_trigger_type_for_turn(
        self,
        task: dict[str, Any],
        agent_id: str,
        prior_target_turns: int,
    ) -> str:
        if self.metric != "idr":
            return "normal"

        idr = task.get("injections", {}).get("idr", {})
        if agent_id != idr.get("target_agent"):
            return "normal"

        trigger_prompts = idr.get("trigger_prompts", [])
        if not trigger_prompts:
            return "normal"
        if prior_target_turns == 0:
            return "low_pressure"
        if len(trigger_prompts) >= 2:
            return "high_pressure"
        return "low_pressure"

    def _augment_user_prompt_for_metric(
        self,
        task: dict[str, Any],
        user_prompt: str,
        trigger_type: str,
    ) -> str:
        """Attach target-turn probes that are part of a metric intervention."""
        if self.metric != "idr" or trigger_type == "normal":
            return user_prompt

        prompts = task.get("injections", {}).get("idr", {}).get("trigger_prompts", [])
        if trigger_type == "high_pressure" and len(prompts) >= 2:
            trigger = prompts[1]
        else:
            trigger = prompts[0] if prompts else ""

        if not trigger:
            return user_prompt
        return f"{user_prompt}\n\n[IDR trigger probe]\n{trigger}"

    def _system_prompt_for_metric(
        self,
        config: dict[str, Any],
        agent_id: str,
        task: dict[str, Any],
    ) -> str:
        system_prompt = config.get("system_prompt", "")
        injections = task.get("injections", {})

        if self.metric == "idr":
            idr = injections.get("idr", {})
            if agent_id == idr.get("target_agent"):
                return build_idr_system_prompt_patch(system_prompt, idr.get("constraint", ""))

        if self.metric == "cpr":
            cpr = injections.get("cpr", {})
            seed_agent = cpr.get("seed_agent") or cpr.get("patient_zero")
            extra_agents = set(cpr.get("extra_agents") or [])
            if agent_id == seed_agent or agent_id in extra_agents:
                return build_cpr_system_prompt_patch(system_prompt, cpr.get("false_fact", ""))
            if cpr.get("ground_truth"):
                return build_cpr_nonseed_system_prompt_patch(
                    system_prompt, cpr.get("ground_truth", "")
                )

        return system_prompt

    def _assemble_parent_context(
        self,
        parents: list[str],
        agent_outputs: dict[str, str],
    ) -> str:
        """Build the user message for a non-root agent from its parents' outputs."""
        if len(parents) == 1:
            return agent_outputs.get(parents[0], "")

        # Multiple parents (converging DAG): label each parent's contribution
        parts = []
        for parent_id in parents:
            output = agent_outputs.get(parent_id, "")
            if output:
                parts.append(f"[From {parent_id}]:\n{output}")
        return "\n\n".join(parts)

    def run_task_file(self, path: str | Path) -> RunResult:
        """Load a task JSON file and run it.

        Args:
            path: Path to a task JSON file (e.g., tasks/TASK-SWE-RTD-001.json).

        Returns:
            RunResult.
        """
        path = Path(path)
        with open(path, encoding="utf-8") as f:
            task = json.load(f)
        return self.run_task(task)

    def run_dataset(self, directory: str | Path) -> list[RunResult]:
        """Run all task JSON files in a directory.

        Args:
            directory: Path to a dataset directory (e.g., tasks/).

        Returns:
            List of RunResult, one per task file.
        """
        directory = Path(directory)
        results = []
        for path in sorted(directory.glob("*.json")):
            logger.info(f"Running task: {path.name}")
            result = self.run_task_file(path)
            results.append(result)
        return results
