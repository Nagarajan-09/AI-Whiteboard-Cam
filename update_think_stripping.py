import base64
import re


path = "app/services/nvidia_service.py"

with open(path, "r", encoding="utf-8") as f:
    content = f.read()

old_method = '''    @staticmethod
    def _clean_mermaid_code(content: str) -> str:
        """
        Remove Markdown code fences, analysis text, and any content
        before the actual Mermaid diagram.
        """

        cleaned = content.strip()

        # Strip everything up to and including a "MERMAID:" marker, if present.
        marker_index = cleaned.upper().find("MERMAID:")
        if marker_index != -1:
            cleaned = cleaned[marker_index + len("MERMAID:"):].strip()

        if cleaned.startswith("```mermaid"):
            cleaned = cleaned[len("```mermaid"):].strip()
        elif cleaned.startswith("```"):
            cleaned = cleaned[3:].strip()

        if cleaned.endswith("```"):
            cleaned = cleaned[:-3].strip()

        mermaid_start = cleaned.find("flowchart")

        if mermaid_start == -1:
            mermaid_start = cleaned.find("graph")

        if mermaid_start > 0:
            cleaned = cleaned[mermaid_start:]

        return cleaned.strip()'''

new_method = '''    @staticmethod
    def _clean_mermaid_code(content: str) -> str:
        """
        Remove Markdown code fences, analysis text, thinking-mode
        <think> blocks, and any content before the actual Mermaid
        diagram.
        """

        cleaned = content.strip()

        # Strip <think>...</think> reasoning blocks some models
        # (e.g. Qwen3.5 in thinking mode) prepend to their response.
        cleaned = re.sub(
            r"<think>.*?</think>",
            "",
            cleaned,
            flags=re.DOTALL | re.IGNORECASE,
        ).strip()

        # Strip everything up to and including a "MERMAID:" marker, if present.
        marker_index = cleaned.upper().find("MERMAID:")
        if marker_index != -1:
            cleaned = cleaned[marker_index + len("MERMAID:"):].strip()

        if cleaned.startswith("```mermaid"):
            cleaned = cleaned[len("```mermaid"):].strip()
        elif cleaned.startswith("```"):
            cleaned = cleaned[3:].strip()

        if cleaned.endswith("```"):
            cleaned = cleaned[:-3].strip()

        mermaid_start = cleaned.find("flowchart")

        if mermaid_start == -1:
            mermaid_start = cleaned.find("graph")

        if mermaid_start > 0:
            cleaned = cleaned[mermaid_start:]

        return cleaned.strip()'''

if old_method not in content:
    print("ERROR: could not find the expected _clean_mermaid_code method to replace.")
    print("No changes were made. Please paste the current file contents back for review.")
else:
    content = content.replace(old_method, new_method)

    # Ensure the "import re" is present near the top imports.
    if "\nimport re\n" not in content and not content.startswith("import re\n"):
        content = content.replace("import base64\n", "import base64\nimport re\n", 1)

    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

    print("Done. _clean_mermaid_code updated with <think> tag stripping, and 'import re' added.")