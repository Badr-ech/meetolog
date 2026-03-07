"use client";

import { useState, useCallback, useRef, useEffect } from "react";
import {
  MeetingArtifacts,
  UserStory,
  Task,
  Decision,
  Blocker,
  ActionItem,
  ActionableTask,
  updateArtifacts,
  getPdfDownloadUrl,
  getJiraExportUrl,
} from "@/lib/api";
import type { BadgeVariant } from "@/types";
import ArtifactBadge, { ConfidenceIndicator } from "./ui/ArtifactBadge";
import styles from "./ArtifactEditor.module.css";

// ---------------------------------------------------------------------------
// Props
// ---------------------------------------------------------------------------

interface ArtifactEditorProps {
  artifacts: MeetingArtifacts;
  jobId: string;
  /** Lift the edited artifacts back to the parent so page-level state stays in sync */
  onArtifactsChange?: (artifacts: MeetingArtifacts) => void;
}

// ---------------------------------------------------------------------------
// Toast helper
// ---------------------------------------------------------------------------

type ToastType = "success" | "error";

function Toast({ message, type }: { message: string; type: ToastType }) {
  return (
    <div
      className={`${styles.toast} ${
        type === "success" ? styles.toastSuccess : styles.toastError
      }`}
    >
      {message}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

interface ValidationErrors {
  [path: string]: string;
}

function validate(artifacts: MeetingArtifacts): ValidationErrors {
  const errors: ValidationErrors = {};

  artifacts.user_stories.forEach((s, i) => {
    if (!s.title.trim()) errors[`user_stories.${i}.title`] = "Title is required";
    if (!s.as_a.trim()) errors[`user_stories.${i}.as_a`] = "Required";
    if (!s.i_want.trim()) errors[`user_stories.${i}.i_want`] = "Required";
    if (!s.so_that.trim()) errors[`user_stories.${i}.so_that`] = "Required";
  });

  artifacts.tasks.forEach((t, i) => {
    if (!t.title.trim()) errors[`tasks.${i}.title`] = "Title is required";
  });

  artifacts.decisions.forEach((d, i) => {
    if (!d.title.trim()) errors[`decisions.${i}.title`] = "Title is required";
    if (!d.description.trim()) errors[`decisions.${i}.description`] = "Description is required";
  });

  artifacts.blockers.forEach((b, i) => {
    if (!b.title.trim()) errors[`blockers.${i}.title`] = "Title is required";
    if (!b.description.trim()) errors[`blockers.${i}.description`] = "Description is required";
  });

  artifacts.execution_tasks.forEach((et, i) => {
    if (!et.title.trim()) errors[`execution_tasks.${i}.title`] = "Title is required";
    if (!et.description.trim()) errors[`execution_tasks.${i}.description`] = "Description is required";
    if (!et.owner_role.trim()) errors[`execution_tasks.${i}.owner_role`] = "Required";
  });

  return errors;
}

// ---------------------------------------------------------------------------
// Utility: deep clone via structured clone (works in all modern browsers)
// ---------------------------------------------------------------------------

function deepClone<T>(obj: T): T {
  return structuredClone(obj);
}

// ---------------------------------------------------------------------------
// Inline field helpers
// ---------------------------------------------------------------------------

function FieldError({ path, errors }: { path: string; errors: ValidationErrors }) {
  if (!errors[path]) return null;
  return <span className={styles.validationError}>{errors[path]}</span>;
}

// ---------------------------------------------------------------------------
// ArtifactEditor Component
// ---------------------------------------------------------------------------

export default function ArtifactEditor({
  artifacts: initialArtifacts,
  jobId,
  onArtifactsChange,
}: ArtifactEditorProps) {
  const [draft, setDraft] = useState<MeetingArtifacts>(() => deepClone(initialArtifacts));
  const [saving, setSaving] = useState(false);
  const [toast, setToast] = useState<{ message: string; type: ToastType } | null>(null);
  const [errors, setErrors] = useState<ValidationErrors>({});
  const prevRef = useRef<MeetingArtifacts>(deepClone(initialArtifacts));
  const toastTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Keep draft in sync when parent provides new artifacts (e.g. after re-poll)
  useEffect(() => {
    setDraft(deepClone(initialArtifacts));
    prevRef.current = deepClone(initialArtifacts);
  }, [initialArtifacts]);

  const showToast = useCallback((message: string, type: ToastType) => {
    setToast({ message, type });
    if (toastTimer.current) clearTimeout(toastTimer.current);
    toastTimer.current = setTimeout(() => setToast(null), 3500);
  }, []);

  // Generic updater – works for any top-level array field
  const updateItem = useCallback(
    <K extends keyof MeetingArtifacts>(
      section: K,
      index: number,
      field: string,
      value: unknown,
    ) => {
      setDraft((prev) => {
        const next = deepClone(prev);
        const arr = next[section] as unknown as Record<string, unknown>[];
        arr[index] = { ...arr[index], [field]: value };
        return next;
      });
    },
    [],
  );

  // Update a top-level scalar field (meeting_title, summary, etc.)
  const updateField = useCallback(
    <K extends keyof MeetingArtifacts>(field: K, value: MeetingArtifacts[K]) => {
      setDraft((prev) => ({ ...prev, [field]: value }));
    },
    [],
  );

  // ---- Save handler with optimistic update ----------------------------------
  const handleSave = useCallback(async () => {
    const validationErrors = validate(draft);
    setErrors(validationErrors);

    if (Object.keys(validationErrors).length > 0) {
      showToast("Fix validation errors before saving", "error");
      return;
    }

    // Snapshot the previous state for rollback
    const rollback = deepClone(prevRef.current);

    // Optimistic update
    prevRef.current = deepClone(draft);
    onArtifactsChange?.(draft);

    setSaving(true);

    try {
      await updateArtifacts(jobId, draft);
      showToast("Changes saved", "success");
    } catch (err) {
      // Revert to the state before the optimistic update
      setDraft(deepClone(rollback));
      prevRef.current = rollback;
      onArtifactsChange?.(rollback);
      showToast(
        err instanceof Error ? err.message : "Failed to save changes",
        "error",
      );
    } finally {
      setSaving(false);
    }
  }, [draft, jobId, onArtifactsChange, showToast]);

  // ---------------------------------------------------------------------------
  // List-item helpers (acceptance_criteria, dependencies, affected_tasks)
  // ---------------------------------------------------------------------------
  const updateListItem = useCallback(
    <K extends keyof MeetingArtifacts>(
      section: K,
      itemIndex: number,
      listField: string,
      listIndex: number,
      value: string,
    ) => {
      setDraft((prev) => {
        const next = deepClone(prev);
        const arr = next[section] as unknown as Record<string, unknown>[];
        const list = [...(arr[itemIndex][listField] as string[])];
        list[listIndex] = value;
        arr[itemIndex] = { ...arr[itemIndex], [listField]: list };
        return next;
      });
    },
    [],
  );

  const addListItem = useCallback(
    <K extends keyof MeetingArtifacts>(section: K, itemIndex: number, listField: string) => {
      setDraft((prev) => {
        const next = deepClone(prev);
        const arr = next[section] as unknown as Record<string, unknown>[];
        const list = [...(arr[itemIndex][listField] as string[]), ""];
        arr[itemIndex] = { ...arr[itemIndex], [listField]: list };
        return next;
      });
    },
    [],
  );

  const removeListItem = useCallback(
    <K extends keyof MeetingArtifacts>(
      section: K,
      itemIndex: number,
      listField: string,
      listIndex: number,
    ) => {
      setDraft((prev) => {
        const next = deepClone(prev);
        const arr = next[section] as unknown as Record<string, unknown>[];
        const list = (arr[itemIndex][listField] as string[]).filter(
          (_, j) => j !== listIndex,
        );
        arr[itemIndex] = { ...arr[itemIndex], [listField]: list };
        return next;
      });
    },
    [],
  );

  // ---------------------------------------------------------------------------
  // Render helpers
  // ---------------------------------------------------------------------------

  const renderListField = (
    section: keyof MeetingArtifacts,
    itemIndex: number,
    listField: string,
    items: string[],
    label: string,
  ) => (
    <div className={styles.fieldFull}>
      <span className={styles.label}>{label}</span>
      <div className={styles.listField}>
        {items.map((item, li) => (
          <div key={li} className={styles.listRow}>
            <input
              className={styles.input}
              value={item}
              onChange={(e) =>
                updateListItem(section, itemIndex, listField, li, e.target.value)
              }
              style={{ flex: 1 }}
            />
            <button
              type="button"
              className={styles.removeBtn}
              onClick={() => removeListItem(section, itemIndex, listField, li)}
              title="Remove"
            >
              ×
            </button>
          </div>
        ))}
        <button
          type="button"
          className={styles.addBtn}
          onClick={() => addListItem(section, itemIndex, listField)}
        >
          + Add
        </button>
      </div>
    </div>
  );

  // ---------------------------------------------------------------------------
  // JSX
  // ---------------------------------------------------------------------------

  return (
    <div className={styles.editor}>
      {toast && <Toast message={toast.message} type={toast.type} />}

      {/* ---- Summary Card ---- */}
      <div className={styles.summaryCard}>
        <div className={styles.summaryHeader}>
          <div style={{ flex: 1 }}>
            <label className={styles.summaryLabel}>Meeting Title</label>
            <input
              className={styles.summaryInput}
              value={draft.meeting_title}
              onChange={(e) => updateField("meeting_title", e.target.value)}
              style={{ width: "100%", fontSize: "1.25rem", fontWeight: 700 }}
            />
            <div style={{ marginTop: "0.5rem" }}>
              <label className={styles.summaryLabel}>Participants (comma-separated)</label>
              <input
                className={styles.summaryInput}
                value={draft.participants.join(", ")}
                onChange={(e) =>
                  updateField(
                    "participants",
                    e.target.value.split(",").map((s) => s.trim()).filter(Boolean),
                  )
                }
                style={{ width: "100%" }}
              />
            </div>
          </div>
          <div style={{ display: "flex", gap: "0.5rem", alignSelf: "flex-start" }}>
            <a
              href={getPdfDownloadUrl(jobId)}
              download
              className="btn btn-primary"
            >
              Download PDF
            </a>
            <a
              href={getJiraExportUrl(jobId)}
              download
              className="btn btn-primary"
              style={{ background: "#7c3aed" }}
            >
              Export to Jira (JSON)
            </a>
          </div>
        </div>
        <div style={{ marginTop: "1rem" }}>
          <label className={styles.summaryLabel}>Summary</label>
          <textarea
            className={styles.summaryTextarea}
            value={draft.summary}
            onChange={(e) => updateField("summary", e.target.value)}
            rows={3}
            style={{ width: "100%" }}
          />
        </div>
      </div>

      {/* ---- User Stories ---- */}
      {draft.user_stories.length > 0 && (
        <div className={styles.section}>
          <h3 className={styles.sectionTitle}>
            User Stories ({draft.user_stories.length})
          </h3>
          {draft.user_stories.map((story: UserStory, i: number) => (
            <div key={story.id} className={`${styles.itemCard} ${styles.storyBorder}`}>
              <div className={styles.fieldRow}>
                <div className={styles.field}>
                  <label className={styles.label}>
                    Title{" "}
                    <ConfidenceIndicator score={story.confidence_score ?? null} />
                  </label>
                  <input
                    className={`${styles.input} ${errors[`user_stories.${i}.title`] ? styles.inputError : ""}`}
                    value={story.title}
                    onChange={(e) => updateItem("user_stories", i, "title", e.target.value)}
                  />
                  <FieldError path={`user_stories.${i}.title`} errors={errors} />
                </div>
                <div className={styles.field} style={{ maxWidth: 120 }}>
                  <label className={styles.label}>Priority</label>
                  <select
                    className={styles.select}
                    value={story.priority}
                    onChange={(e) => updateItem("user_stories", i, "priority", e.target.value)}
                  >
                    <option value="low">Low</option>
                    <option value="medium">Medium</option>
                    <option value="high">High</option>
                    <option value="critical">Critical</option>
                  </select>
                </div>
                <div className={styles.field} style={{ maxWidth: 80 }}>
                  <label className={styles.label}>Points</label>
                  <input
                    className={styles.input}
                    type="number"
                    min={0}
                    value={story.story_points ?? ""}
                    onChange={(e) =>
                      updateItem(
                        "user_stories",
                        i,
                        "story_points",
                        e.target.value ? Number(e.target.value) : null,
                      )
                    }
                  />
                </div>
              </div>
              <div className={styles.fieldRow}>
                <div className={styles.field}>
                  <label className={styles.label}>As a…</label>
                  <input
                    className={`${styles.input} ${errors[`user_stories.${i}.as_a`] ? styles.inputError : ""}`}
                    value={story.as_a}
                    onChange={(e) => updateItem("user_stories", i, "as_a", e.target.value)}
                  />
                  <FieldError path={`user_stories.${i}.as_a`} errors={errors} />
                </div>
                <div className={styles.field}>
                  <label className={styles.label}>I want…</label>
                  <input
                    className={`${styles.input} ${errors[`user_stories.${i}.i_want`] ? styles.inputError : ""}`}
                    value={story.i_want}
                    onChange={(e) => updateItem("user_stories", i, "i_want", e.target.value)}
                  />
                  <FieldError path={`user_stories.${i}.i_want`} errors={errors} />
                </div>
                <div className={styles.field}>
                  <label className={styles.label}>So that…</label>
                  <input
                    className={`${styles.input} ${errors[`user_stories.${i}.so_that`] ? styles.inputError : ""}`}
                    value={story.so_that}
                    onChange={(e) => updateItem("user_stories", i, "so_that", e.target.value)}
                  />
                  <FieldError path={`user_stories.${i}.so_that`} errors={errors} />
                </div>
              </div>
              {renderListField(
                "user_stories",
                i,
                "acceptance_criteria",
                story.acceptance_criteria,
                "Acceptance Criteria",
              )}
            </div>
          ))}
        </div>
      )}

      {/* ---- Tasks ---- */}
      {draft.tasks.length > 0 && (
        <div className={styles.section}>
          <h3 className={styles.sectionTitle}>Tasks ({draft.tasks.length})</h3>
          {draft.tasks.map((task: Task, i: number) => (
            <div key={task.id} className={styles.itemCard}>
              <div className={styles.fieldRow}>
                <div className={styles.field}>
                  <label className={styles.label}>
                    Title{" "}
                    <ConfidenceIndicator score={task.confidence_score ?? null} />
                  </label>
                  <input
                    className={`${styles.input} ${errors[`tasks.${i}.title`] ? styles.inputError : ""}`}
                    value={task.title}
                    onChange={(e) => updateItem("tasks", i, "title", e.target.value)}
                  />
                  <FieldError path={`tasks.${i}.title`} errors={errors} />
                </div>
                <div className={styles.field} style={{ maxWidth: 120 }}>
                  <label className={styles.label}>Priority</label>
                  <select
                    className={styles.select}
                    value={task.priority}
                    onChange={(e) => updateItem("tasks", i, "priority", e.target.value)}
                  >
                    <option value="low">Low</option>
                    <option value="medium">Medium</option>
                    <option value="high">High</option>
                    <option value="critical">Critical</option>
                  </select>
                </div>
                <div className={styles.field} style={{ maxWidth: 140 }}>
                  <label className={styles.label}>Status</label>
                  <select
                    className={styles.select}
                    value={task.status}
                    onChange={(e) => updateItem("tasks", i, "status", e.target.value)}
                  >
                    <option value="todo">Todo</option>
                    <option value="in_progress">In Progress</option>
                    <option value="blocked">Blocked</option>
                    <option value="done">Done</option>
                  </select>
                </div>
              </div>
              <div className={styles.fieldRow}>
                <div className={styles.fieldFull}>
                  <label className={styles.label}>Description</label>
                  <textarea
                    className={styles.textarea}
                    value={task.description}
                    onChange={(e) => updateItem("tasks", i, "description", e.target.value)}
                    rows={2}
                  />
                </div>
              </div>
              <div className={styles.fieldRow}>
                <div className={styles.field}>
                  <label className={styles.label}>Assignee</label>
                  <input
                    className={styles.input}
                    value={task.assignee ?? ""}
                    onChange={(e) =>
                      updateItem("tasks", i, "assignee", e.target.value || null)
                    }
                  />
                </div>
                <div className={styles.field} style={{ maxWidth: 160 }}>
                  <label className={styles.label}>Due Date</label>
                  <input
                    className={styles.input}
                    value={task.due_date ?? ""}
                    onChange={(e) =>
                      updateItem("tasks", i, "due_date", e.target.value || null)
                    }
                    placeholder="YYYY-MM-DD"
                  />
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* ---- Decisions ---- */}
      {draft.decisions.length > 0 && (
        <div className={styles.section}>
          <h3 className={styles.sectionTitle}>
            Decisions ({draft.decisions.length})
          </h3>
          {draft.decisions.map((d: Decision, i: number) => (
            <div key={d.id} className={`${styles.itemCard} ${styles.decisionBorder}`}>
              <div className={styles.fieldRow}>
                <div className={styles.field}>
                  <label className={styles.label}>
                    Title{" "}
                    <ConfidenceIndicator score={d.confidence_score ?? null} />
                  </label>
                  <input
                    className={`${styles.input} ${errors[`decisions.${i}.title`] ? styles.inputError : ""}`}
                    value={d.title}
                    onChange={(e) => updateItem("decisions", i, "title", e.target.value)}
                  />
                  <FieldError path={`decisions.${i}.title`} errors={errors} />
                </div>
                <div className={styles.field} style={{ maxWidth: 180 }}>
                  <label className={styles.label}>Made By</label>
                  <input
                    className={styles.input}
                    value={d.made_by ?? ""}
                    onChange={(e) =>
                      updateItem("decisions", i, "made_by", e.target.value || null)
                    }
                  />
                </div>
              </div>
              <div className={styles.fieldRow}>
                <div className={styles.fieldFull}>
                  <label className={styles.label}>Description</label>
                  <textarea
                    className={`${styles.textarea} ${errors[`decisions.${i}.description`] ? styles.inputError : ""}`}
                    value={d.description}
                    onChange={(e) => updateItem("decisions", i, "description", e.target.value)}
                    rows={2}
                  />
                  <FieldError path={`decisions.${i}.description`} errors={errors} />
                </div>
              </div>
              <div className={styles.fieldRow}>
                <div className={styles.fieldFull}>
                  <label className={styles.label}>Rationale</label>
                  <textarea
                    className={styles.textarea}
                    value={d.rationale}
                    onChange={(e) => updateItem("decisions", i, "rationale", e.target.value)}
                    rows={2}
                  />
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* ---- Blockers ---- */}
      {draft.blockers.length > 0 && (
        <div className={styles.section}>
          <h3 className={styles.sectionTitle}>
            Blockers ({draft.blockers.length})
          </h3>
          {draft.blockers.map((b: Blocker, i: number) => (
            <div key={b.id} className={`${styles.itemCard} ${styles.blockerBorder}`}>
              <div className={styles.fieldRow}>
                <div className={styles.field}>
                  <label className={styles.label}>
                    Title{" "}
                    <ConfidenceIndicator score={b.confidence_score ?? null} />
                  </label>
                  <input
                    className={`${styles.input} ${errors[`blockers.${i}.title`] ? styles.inputError : ""}`}
                    value={b.title}
                    onChange={(e) => updateItem("blockers", i, "title", e.target.value)}
                  />
                  <FieldError path={`blockers.${i}.title`} errors={errors} />
                </div>
                <div className={styles.field} style={{ maxWidth: 180 }}>
                  <label className={styles.label}>Owner</label>
                  <input
                    className={styles.input}
                    value={b.owner ?? ""}
                    onChange={(e) =>
                      updateItem("blockers", i, "owner", e.target.value || null)
                    }
                  />
                </div>
              </div>
              <div className={styles.fieldRow}>
                <div className={styles.fieldFull}>
                  <label className={styles.label}>Description</label>
                  <textarea
                    className={`${styles.textarea} ${errors[`blockers.${i}.description`] ? styles.inputError : ""}`}
                    value={b.description}
                    onChange={(e) => updateItem("blockers", i, "description", e.target.value)}
                    rows={2}
                  />
                  <FieldError path={`blockers.${i}.description`} errors={errors} />
                </div>
              </div>
              <div className={styles.fieldRow}>
                <div className={styles.fieldFull}>
                  <label className={styles.label}>Resolution Plan</label>
                  <textarea
                    className={styles.textarea}
                    value={b.resolution_plan}
                    onChange={(e) =>
                      updateItem("blockers", i, "resolution_plan", e.target.value)
                    }
                    rows={2}
                  />
                </div>
              </div>
              {renderListField(
                "blockers",
                i,
                "affected_tasks",
                b.affected_tasks,
                "Affected Tasks",
              )}
            </div>
          ))}
        </div>
      )}

      {/* ---- Action Items ---- */}
      {draft.action_items.length > 0 && (
        <div className={styles.section}>
          <h3 className={styles.sectionTitle}>
            Action Items ({draft.action_items.length})
          </h3>
          {draft.action_items.map((item: ActionItem, i: number) => (
            <div key={item.id} className={styles.itemCard}>
              <div className={styles.fieldRow}>
                <div className={styles.fieldFull}>
                  <label className={styles.label}>Description</label>
                  <input
                    className={styles.input}
                    value={item.description}
                    onChange={(e) =>
                      updateItem("action_items", i, "description", e.target.value)
                    }
                  />
                </div>
              </div>
              <div className={styles.fieldRow}>
                <div className={styles.field}>
                  <label className={styles.label}>Assignee</label>
                  <input
                    className={styles.input}
                    value={item.assignee ?? ""}
                    onChange={(e) =>
                      updateItem("action_items", i, "assignee", e.target.value || null)
                    }
                  />
                </div>
                <div className={styles.field} style={{ maxWidth: 160 }}>
                  <label className={styles.label}>Due Date</label>
                  <input
                    className={styles.input}
                    value={item.due_date ?? ""}
                    onChange={(e) =>
                      updateItem("action_items", i, "due_date", e.target.value || null)
                    }
                    placeholder="YYYY-MM-DD"
                  />
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* ---- Execution Tasks ---- */}
      {draft.execution_tasks.length > 0 && (
        <div className={styles.section}>
          <h3 className={styles.sectionTitle}>
            Execution Tasks ({draft.execution_tasks.length})
          </h3>
          {draft.execution_tasks.map((et: ActionableTask, i: number) => {
            const sourceVariant: BadgeVariant =
              et.task_source?.toLowerCase() === "inferred"
                ? "inferred"
                : et.task_source?.toLowerCase() === "explicit"
                  ? "explicit"
                  : "default";

            return (
            <div key={i} className={`${styles.itemCard} ${styles.executionBorder}`}>
              <div className={styles.fieldRow}>
                <div className={styles.field}>
                  <label className={styles.label}>
                    Title{" "}
                    <ArtifactBadge
                      variant={sourceVariant}
                      label={et.task_source ?? "Unknown"}
                    />
                    {" "}
                    <ConfidenceIndicator score={et.confidence_score ?? null} />
                  </label>
                  <input
                    className={`${styles.input} ${errors[`execution_tasks.${i}.title`] ? styles.inputError : ""}`}
                    value={et.title}
                    onChange={(e) =>
                      updateItem("execution_tasks", i, "title", e.target.value)
                    }
                  />
                  <FieldError path={`execution_tasks.${i}.title`} errors={errors} />
                </div>
                <div className={styles.field} style={{ maxWidth: 120 }}>
                  <label className={styles.label}>Priority</label>
                  <select
                    className={styles.select}
                    value={et.priority}
                    onChange={(e) =>
                      updateItem("execution_tasks", i, "priority", e.target.value)
                    }
                  >
                    <option value="High">High</option>
                    <option value="Medium">Medium</option>
                    <option value="Low">Low</option>
                  </select>
                </div>
                <div className={styles.field} style={{ maxWidth: 120 }}>
                  <label className={styles.label}>Source</label>
                  <select
                    className={styles.select}
                    value={et.task_source}
                    onChange={(e) =>
                      updateItem("execution_tasks", i, "task_source", e.target.value)
                    }
                  >
                    <option value="Explicit">Explicit</option>
                    <option value="Inferred">Inferred</option>
                  </select>
                </div>
              </div>
              <div className={styles.fieldRow}>
                <div className={styles.fieldFull}>
                  <label className={styles.label}>Description</label>
                  <textarea
                    className={`${styles.textarea} ${errors[`execution_tasks.${i}.description`] ? styles.inputError : ""}`}
                    value={et.description}
                    onChange={(e) =>
                      updateItem("execution_tasks", i, "description", e.target.value)
                    }
                    rows={2}
                  />
                  <FieldError path={`execution_tasks.${i}.description`} errors={errors} />
                </div>
              </div>
              <div className={styles.fieldRow}>
                <div className={styles.field}>
                  <label className={styles.label}>Owner / Role</label>
                  <input
                    className={`${styles.input} ${errors[`execution_tasks.${i}.owner_role`] ? styles.inputError : ""}`}
                    value={et.owner_role}
                    onChange={(e) =>
                      updateItem("execution_tasks", i, "owner_role", e.target.value)
                    }
                  />
                  <FieldError path={`execution_tasks.${i}.owner_role`} errors={errors} />
                </div>
              </div>
              {renderListField(
                "execution_tasks",
                i,
                "dependencies",
                et.dependencies,
                "Dependencies",
              )}
            </div>
            );
          })}
        </div>
      )}

      {/* ---- Raw JSON Toggle ---- */}
      <details className={styles.section}>
        <summary style={{ cursor: "pointer", fontWeight: 600, color: "#6b7280" }}>
          View Raw JSON
        </summary>
        <pre
          style={{
            marginTop: "1rem",
            padding: "1rem",
            background: "#1f2937",
            color: "#10b981",
            borderRadius: 8,
            fontSize: "0.75rem",
            overflowX: "auto",
            maxHeight: 400,
          }}
        >
          {JSON.stringify(draft, null, 2)}
        </pre>
      </details>

      {/* ---- Save Bar ---- */}
      <div className={styles.saveBar}>
        {saving && <span className={styles.savingLabel}>Saving…</span>}
        <button
          className={styles.saveBtn}
          onClick={handleSave}
          disabled={saving}
        >
          {saving ? "Saving…" : "Save Changes"}
        </button>
      </div>
    </div>
  );
}
