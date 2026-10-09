# AGENTS.md: Developer & Agent Operating Guidelines

This guide defines strict operating rules and constraints for autonomous agents working on **HypeLens**.

---

## 1. Pre-Implementation Protocol
- **Read Specifications First:** Always read [PROJECT.md](file:///C:/Users/Admin/.gemini/antigravity/scratch/hypelens/PROJECT.md) and [TASKS.md](file:///C:/Users/Admin/.gemini/antigravity/scratch/hypelens/TASKS.md) before making modifications or starting a task.
- **Strict Dependency Order:** Do not work on a task until all preceding dependent tasks are completed and verified by tests.
- **Update Task States Accurately:** Keep task progress updated in `TASKS.md`. Never mark a task as completed without executing its verification command.

---

## 2. Mandatory Verification & Completion Reporting
- **Execute Task-Specific Verification Command:** Before marking any task complete, the agent must run the exact **Verification Command** defined for that task in `TASKS.md`.
- **No Fabricated Test Results:** Prohibit claiming that an unexecuted test passed or inventing simulated command outputs. Actual test command execution and genuine test output are mandatory prerequisites for task completion.
- **Concise Completion Reports:** After each task, report modified files, actual test execution output, and any blocked items concisely.

---

## 3. Scope Discipline & Strict Prohibitions
1. **No Unrelated Refactoring:** Do not reorganize existing project structures, rename modules, or rewrite unrelated files unless explicitly tasked.
2. **No Paid LLM APIs:** Do not introduce dependencies on OpenAI, Anthropic, or other paid cloud APIs. All cognitive extraction must run on local **Gemma 4 via Ollama** (with offline mock fallbacks for testing).
3. **No Invented Real-World Data:** Do not hallucinate or manufacture false financial histories or real-world social posts. Use curated fixtures or verified historical records.
4. **No Unsupported Stock Predictions or Guarantees:** Predictions are probabilistic decision-support signals only; never make claims guaranteeing future price moves or returns.
5. **No Automatic Trading or Personalization:** Never implement trade execution brokers, portfolio management, or user risk profiling.

---

## 4. Required Edge Case & Failure Handling
- **Missing Timestamps:** Social posts or news items lacking reliable timestamps must be explicitly quarantined or flagged with a fallback imputation policy; never mix indeterminate timestamps into leakage-sensitive time windows.
- **Epistemic Uncertainty:** Do not force speculative social posts into binary truths. Explicitly tag unverified claims as `Rumour/Unverified` with quantified uncertainty.
- **Model Inference Failures:** Implement robust fallbacks (retries, regex repairs, mock extractor fallbacks) for Ollama connection drops, malformed JSON outputs, or context length exceptions.
