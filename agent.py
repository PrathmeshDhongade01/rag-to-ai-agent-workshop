import os
import json
import re

from dotenv import load_dotenv
from google import genai

from tools import rag_search, create_report
from email_tool import send_email


# ==========================================
# LOAD ENVIRONMENT
# ==========================================

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY is missing from .env")


# ==========================================
# GEMINI CLIENT
# ==========================================

client = genai.Client(api_key=api_key)


# ==========================================
# AVAILABLE TOOLS
# ==========================================

TOOLS = {
    "rag_search": rag_search,
    "create_report": create_report,
    "send_email": send_email
}


# ==========================================
# ASK GEMINI
# ==========================================

def ask_gemini(prompt):

    interaction = client.interactions.create(
        model="gemini-3.8-flash",
        input=prompt
    )

    return interaction.output_text


# ==========================================
# EXTRACT JSON
# ==========================================

def extract_json(text):

    try:
        return json.loads(text)

    except json.JSONDecodeError:
        pass

    match = re.search(
        r"```json\s*(.*?)\s*```",
        text,
        re.DOTALL
    )

    if match:

        return json.loads(
            match.group(1)
        )

    match = re.search(
        r"\{.*\}",
        text,
        re.DOTALL
    )

    if match:

        return json.loads(
            match.group(0)
        )

    raise ValueError(
        "Gemini did not return valid JSON."
    )


# ==========================================
# AGENT PLANNER
# ==========================================

def create_plan(user_request):

    print("\n[Agent] Creating plan...")

    prompt = f"""
You are an AI agent for KBTCOE college.

You have these tools:

1. rag_search
   Search the KBTCOE knowledge base.

2. create_report
   Create a text report.

3. send_email
   Send an email.

You can use MULTIPLE tools in sequence.

For example, if the user says:

"Create a report about KBTCOE and email it to me."

The correct plan is:

1. rag_search
2. create_report
3. send_email

Return ONLY valid JSON.

Format:

{{
    "actions": [
        {{
            "tool": "rag_search",
            "arguments": {{
                "query": "..."
            }}
        }},
        {{
            "tool": "create_report",
            "arguments": {{
                "title": "...",
                "content": "{{previous_result}}"
            }}
        }},
        {{
            "tool": "send_email",
            "arguments": {{
                "to_email": "me",
                "subject": "...",
                "body": "{{previous_result}}"
            }}
        }}
    ]
}}

Important:

- Use rag_search when information from the college document is required.
- Use create_report when the user asks for a report.
- Use send_email when the user asks to send an email.
- If the user says "email it to me", use "me" as to_email.
- "{{previous_result}}" means the result returned by the previous tool.
- Do not invent information.

User request:

{user_request}
"""

    response = ask_gemini(prompt)

    return extract_json(response)


# ==========================================
# EXECUTE PLAN
# ==========================================

def run_agent(user_request):

    plan = create_plan(
        user_request
    )

    actions = plan.get(
        "actions",
        []
    )

    if not actions:

        return "No actions were planned."


    previous_result = ""

    print(
        f"\n[Agent] Plan contains {len(actions)} action(s)."
    )


    # ======================================
    # EXECUTE EACH ACTION
    # ======================================

    for number, action in enumerate(
        actions,
        start=1
    ):

        tool_name = action["tool"]

        arguments = action.get(
            "arguments",
            {}
        )


        print(
            f"\n[Agent] Step {number}: {tool_name}"
        )


        # ----------------------------------
        # Replace previous result
        # ----------------------------------

        for key, value in arguments.items():

            if isinstance(value, str):

                value = value.replace(
                    "{{previous_result}}",
                    previous_result
                )

                arguments[key] = value


        # ----------------------------------
        # "me" = sender email
        # ----------------------------------

        if tool_name == "send_email":

            if arguments.get("to_email") == "me":

                arguments["to_email"] = os.getenv(
                    "GMAIL_ADDRESS"
                )


        # ----------------------------------
        # Validate tool
        # ----------------------------------

        if tool_name not in TOOLS:

            print(
                f"[Agent] Unknown tool: {tool_name}"
            )

            continue


        # ----------------------------------
        # Execute
        # ----------------------------------

        print(
            f"[Tool] Running {tool_name}..."
        )

        tool_function = TOOLS[tool_name]

        previous_result = tool_function(
            **arguments
        )

        print(
            "[Tool] Completed."
        )


    return previous_result


# ==========================================
# RUN
# ==========================================

if __name__ == "__main__":

    print(
        "\n================================"
    )

    print(
        "       KBTCOE AI AGENT"
    )

    print(
        "================================"
    )

    user_request = input(
        "\nWhat would you like me to do? "
    )

    try:

        result = run_agent(
            user_request
        )

        print(
            "\n========== AGENT RESPONSE ==========\n"
        )

        print(result)

    except Exception as e:

        print(
            f"\nAgent error: {e}"
        )