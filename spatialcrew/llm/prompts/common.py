"""Shared prompt sections used across all agents."""

COORDINATE_SYSTEM_SECTION = """## Coordinate System
- Image coordinates: (0,0) is top-left, x increases right, y increases down.
- 3D world coordinates: right-handed, y-up convention unless stated otherwise.
- When the reconstruction tool provides extrinsics, world coordinates are defined by the first frame's camera.
"""

WEVA_FORMAT_SECTION = """## Response Format (WEVA Protocol)
You operate in a Write-Execute-Verify-Adjust loop. Each step must follow this format:

### Purpose
[Brief description of what this step aims to accomplish]

### Reasoning
[Your analysis and reasoning for the approach]

### Code
```python
# Your executable Python code here
```

### Verification
[After seeing execution results, verify the output against expected specifications]

If verification fails, describe the issue and adjust your approach in the next step.
"""

RETURN_ANSWER_SECTION = """## Submitting Results
When you have completed your task and verified the results, submit using:
```python
ReturnAnswer(answer="your_answer", reasoning="brief explanation")
```
Only call ReturnAnswer when you are confident in your result. If uncertain, continue refining.
"""

CODE_RULES_SECTION = """## Code Rules
- Write clean, self-contained Python code in each step.
- Use numpy, scipy, and matplotlib for computation and visualization.
- Always handle edge cases (empty masks, zero division, etc.).
- Use `feedback.show(image)` to visually inspect intermediate results.
- Variables persist across steps within the same session.
- Do NOT use `print()` for large arrays; summarize key values instead.
"""

EVIDENCE_HIERARCHY_SECTION = """## Evidence Hierarchy
When reasoning about spatial properties, prioritize evidence as follows:
1. **Direct measurement** from 3D reconstruction (most reliable)
2. **Geometric computation** from depth maps and camera parameters
3. **Visual estimation** from VLM queries (least reliable, use as sanity check)

Always cross-check VLM-based estimates against tool-based measurements when possible.
"""
