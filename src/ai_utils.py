import requests
import json

# The local endpoint for the Ollama API
OLLAMA_ENDPOINT = "http://localhost:11434/api/generate"

def generate_report_conclusion(selections, metrics, prediction):
    """
    Uses a local Ollama Llama 3 model to generate a dynamic conclusion for the report.
    """
    try:
        # Create the prompt for the model
        prompt = f"""
        You are a professional political data analyst. Your task is to write a concise, insightful conclusion for an automated election report.
        Based on the data provided below, generate a 2-3 paragraph summary.

        **Report Context:**
        - Election Year Analyzed: {selections['year']}
        - State Analyzed: {selections['state']}

        **Key Metrics for this selection:**
        - Total Votes Cast: {metrics['total_votes']}
        - Number of Constituencies: {metrics['total_constituencies']}
        - Number of Candidates: {metrics['total_candidates']}

        **Predictive Model Analysis:**
        - A prediction was run for a hypothetical candidate.
        - Prediction Inputs: {prediction.get('inputs', 'N/A')}
        - Predicted Vote Share: {prediction.get('result', 0.0):.2f}%

        **Your Task:**
        Write a professional "Conclusion" section for the PDF report. Analyze the provided metrics and prediction.
        Keep the tone objective and data-driven. Do not mention that you are an AI.
        """

        # Data payload for the Ollama API
        payload = {
            "model": "llama3",
            "prompt": prompt,
            "stream": False
        }

        # Send the request to the local Ollama server
        response = requests.post(OLLAMA_ENDPOINT, json=payload, timeout=120)
        response.raise_for_status()

        response_data = response.json()
        return response_data.get("response", "Error: Could not parse AI response.")

    except requests.exceptions.ConnectionError:
        return (
            "Conclusion could not be generated. \n"
            "Error: Connection to Ollama failed. Please make sure the Ollama application is running on your machine."
        )
    except Exception as e:
        return (
            f"An error occurred with the Ollama API: {e}. "
            "This section typically provides a dynamic analysis of the dashboard's findings."
        )