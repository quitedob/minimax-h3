---
name: h3-prompt-writing
description: Rewrite a plain text idea into a MiniMax H3 video-generation prompt (T2VA, I2VA, FL2VA, L2VA, Ref2VA), then either auto-fill it into the matching ComfyUI H3 workflow (t2v/i2v/r2v) or hand it back for review. Use when the user wants to turn text into an H3 video prompt and optionally inject it into the workflow JSON.
---

# H3 Prompt Writing

Turns a raw idea into a finished MiniMax H3 prompt and injects it into the ComfyUI
text-to-video workflow. There are two output modes: **auto** (write straight into the
workflow) or **review** (show the prompt, let the user check or edit it, then write).

## End-to-end flow (run in this order)

### 1. Get the raw input
- If the user passed text after `/h3-prompt-writing`, use it.
- Otherwise ask: "What do you want the video to be about?" and wait for their answer.

### 2. Pick the input mode
- **T2VA** (text-to-video) is the default when the user gives text only.
- **I2VA / FL2VA / L2VA** when they provide a first frame and/or last frame image.
- **Ref2VA** when they provide full references (images + video + audio).

Say in one line which mode you picked before writing.

### 3. Write the final prompt
Follow `references/base-en.txt` (T2VA/I2VA/FL2VA/L2VA) or `references/ref-en.txt`
(Ref2VA). Preserve exact field names, section order, labels, and timing notation.

### 4. Show the final prompt
Always print the complete final prompt in a fenced block so the user can read it.

### 5. Choose the output mode (ask the user)
Ask: **"Auto-fill the workflow, or review first?"**
- **auto** → go to step 6.
- **review** → wait for the user to approve or edit the prompt, then use the final
  (possibly edited) text in step 6.

### 6. Write the prompt into the workflow (auto, or after review approval)
Pick the workflow from the mode: **T2VA → `t2v`**, **I2VA/FL2VA/L2VA → `i2v`**,
**Ref2VA → `r2v`**. Write the final prompt to a temp file, then run the helper script
(UTF-8 safe):

```bash
ComfyUI_windows_portable/python_embeded/python.exe \
  .claude/skills/h3-prompt-writing/scripts/update_workflow_prompt.py set \
  --workflow t2v --prompt-file <temp-file>
```

`--workflow` takes a preset (`t2v` | `i2v` | `r2v`) or a path. The script updates
every prompt-bearing slot for that workflow shape (subgraph-exposed prompt + inner
`MiniMaxH3ImageToVideo` for `t2v`/`i2v`, or the `PrimitiveStringMultiline` for `r2v`).

Confirm the update succeeded, then tell the user to:
1. open ComfyUI (run `run_nvidia_gpu.bat` in `ComfyUI_windows_portable/`),
2. load the matching workflow (`video_minimax_h3_t2v.json` / `..._i2v.json` / `..._r2v.json`),
3. press **Run**.

## Prompt-writing rules

### Input modes
- **T2VA**: build the full audiovisual timeline from text.
- **I2VA**: start from the first frame and develop forward from it.
- **FL2VA**: describe the continuous path between the first and last frames.
- **L2VA**: infer a plausible opening and converge to the supplied last frame.
- **Ref2VA**: full-reference rewrite with six sections.

Use `integrated_multimodal_description`, `overall_soundscape`, and `non_diegetic_music`
in the order shown in `references/base-en.txt`. Ref2VA uses `subject_definitions`,
`summary`, `retention_analysis`, `detailed_description`, `overall_soundscape`, and
`non_diegetic_music` in that order (see `references/ref-en.txt`).

### Output rules
- Write rewrite sections in English; preserve dialogue, lyrics, and visible scene text
  in their original language.
- Describe each shot by composition, subjects, environment, actions, camera, sound, and
  the exact point where referenced content appears.
- Avoid plot summaries, unresolved reference labels, and timing that does not match the
  requested duration.

## Notes
- MiniMax H3 has no dedicated "Text to Video" node: T2V is `MiniMaxH3ImageToVideo` with
  `first_frame` / `last_frame` left empty, so the prompt alone drives the generation.
  The `i2v` workflow feeds `first_frame` (and optionally `last_frame`) into the same node;
  `r2v` uses `MiniMaxH3ReferenceToVideo` with a `PrimitiveStringMultiline` prompt.
- To read the current prompt instead of setting it:
  `... update_workflow_prompt.py get --workflow t2v` (same `t2v|i2v|r2v` presets).
