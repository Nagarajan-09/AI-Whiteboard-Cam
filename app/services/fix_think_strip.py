path = "app/services/nvidia_service.py"

with open(path, "r", encoding="utf-8") as f:
    content = f.read()

old = '''    @staticmethod
    def _clean_mermaid_code(content: str) -> str:
        """
        Remove Markdown code fences, analysis text, and any content
        before the actual Mermaid diagram.
        """

        cleaned = content.strip()
'''

new = '''    @staticmethod
    def _clean_mermaid_code(content: str) -> str:
        """
        Remove Markdown code fences, analysis text, thinking-mode
        <think> blocks, and any content before the actual Mermaid
        diagram.
        """

        cleaned = content.strip()

        import re
        cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL | re.IGNORECASE).strip()
'''

if old not in content:
    print("ERROR: expected block not found. No changes made.")
else:
    content = content.replace(old, new, 1)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print("SUCCESS: think-tag stripping added.")

with open(path, "r", encoding="utf-8") as f:
    final = f.read()

print("Verification -> think stripping present:", "think" in final.lower())
print("Verification -> import re present:", "import re" in final)