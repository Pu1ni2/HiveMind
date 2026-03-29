 Ready to code?                                                                                                                                                        
                                                                                                                                                                       
 Here is Claude's plan:                                                                                                                                                
╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌
 Plan: Add MMA Memory System to Clawforge Pipeline                                                                                                                   

 Context

 Clawforge's run_pipeline() is stateless — each call starts fresh with no awareness of prior tasks. We want to add MMA's topic segmentation + compression +
 persistence so the pipeline remembers past runs and can use that context in future ones. This operates at the pipeline level only.

 Concept Mapping

 ┌────────────────────────────┬────────────────────────────────────────────┐
 │            MMA             │           Clawforge Integration            │
 ├────────────────────────────┼────────────────────────────────────────────┤
 │ User message               │ One run_pipeline(task) call                │
 ├────────────────────────────┼────────────────────────────────────────────┤
 │ Segment (topic + messages) │ PipelineSegment (topic + TaskRecords)      │
 ├────────────────────────────┼────────────────────────────────────────────┤
 │ ChatPipeline.step()        │ ForgeSession.run() wrapping run_pipeline() │
 ├────────────────────────────┼────────────────────────────────────────────┤
 │ Compressed summary         │ Summary of past task results               │
 ├────────────────────────────┼────────────────────────────────────────────┤
 │ TopicMap                   │ PipelineTopicMap persisted to JSON         │
 └────────────────────────────┴────────────────────────────────────────────┘

 New Files (4)

 1. clawforge/memory_models.py (~60 lines)

 Adapted from mma/models.py:
 - TaskRecord dataclass: task, final_output, requirements_summary, agent_count, timestamp + token_estimate() + to_dict()/from_dict()
 - PipelineSegment dataclass: topic, records: list[TaskRecord], compressed, summary + token_estimate() + serialization
 - PipelineTopicMap class: same interface as MMA's TopicMap — start_new_topic(), should_compress(), compress_oldest(), get_all_segments_for_context() +
 to_dict()/from_dict() for persistence

 2. clawforge/memory_prompts.py (~40 lines)

 Two prompt templates:
 - TASK_TOPIC_CLASSIFIER_PROMPT: given current topic + recent tasks + new task → {"same_topic": bool, "topic_label": "..."}
 - TASK_SEGMENT_SUMMARY_PROMPT: given a segment's task/output pairs → 2-3 sentence summary preserving key deliverables

 Separate file from prompts.py (441 lines) to keep concerns clean.

 3. clawforge/memory.py (~80 lines)

 MemoryManager class with three methods:
 - classify_topic(current_topic, recent_records, new_task) → (same_topic, topic_label) — uses call_llm_json()
 - summarize_segment(segment) → summary string — uses call_llm()
 - build_memory_context(topic_map) → string for injection — compressed segments as summaries, uncompressed as full task/output pairs

 4. clawforge/session.py (~90 lines)

 ForgeSession class — the main integration point:
 - __init__(memory_path=None): loads topic map from JSON (or creates fresh)
 - run(task, tool_executor=None):
   a. Classify topic
   b. Handle topic change + compression if needed
   c. Build memory context string
   d. Call run_pipeline(task, tool_executor, memory_context=context)
   e. Record TaskRecord in current segment
   f. Save to JSON
   g. Return pipeline result
 - _load() / _save(): JSON file I/O
 - reset(): clear memory

 Modified Files (2)

 5. clawforge/pipeline.py (~4 lines changed)

 - Add memory_context: str = "" parameter to run_pipeline()
 - Pass it as input_context to the Phase 1-2 run_debate() call (line 33-38)
 - Fully backward compatible — default "" preserves current behavior

 6. clawforge/config.py (3 lines added)

 MAX_ACTIVE_SEGMENTS = 3
 MEMORY_FILE = ".clawforge_memory.json"
 MEMORY_MODEL = DEBATE_MODEL

 Untouched Files

 - debate.py — already has input_context param, no changes needed
 - dynamic_agent.py — sub-agents remain stateless
 - sub_agent.py — unchanged
 - llm_client.py — reused as-is
 - prompts.py — unchanged

 Usage

 # Stateless (unchanged, still works):
 result = run_pipeline(task)

 # With memory:
 session = ForgeSession()
 result = session.run("Build a REST API for users")
 result2 = session.run("Add authentication to the API")  # knows about prior work

 Implementation Order

 1. config.py — add constants
 2. memory_models.py — dataclasses + topic map
 3. memory_prompts.py — two templates
 4. memory.py — MemoryManager
 5. pipeline.py — add memory_context param
 6. session.py — ForgeSession wrapper

 Verification

 - Call ForgeSession.run() twice with related tasks, verify second run's Phase 1-2 debate references the first task's context
 - Call with unrelated tasks, verify topic classification creates a new segment
 - Run enough tasks to trigger compression, verify old segments get summarized
 - Check .clawforge_memory.json has correct structure after runs
 - Verify run_pipeline() still works standalone without memory