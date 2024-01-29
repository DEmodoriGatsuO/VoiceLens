import azure.functions as func
import os
import openai
import logging

app = func.FunctionApp(http_auth_level=func.AuthLevel.FUNCTION)

@app.route(route="voicelens", methods=['POST'])
def voicelens(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    messange = req.params.get('messange')
    if not messange:
        try:
            req_body = req.get_json()
        except ValueError:
            pass
        else:
            messange = req_body.get('messange')

    if messange:
        return func.HttpResponse(get_chat_completion(messange))
    else:
        return func.HttpResponse(
             "This HTTP triggered function executed successfully. Pass a name in the query string or in the request body for a personalized response.",
             status_code=200
        )

def get_chat_completion(message_text):
    # Configure the OpenAI API settings
    openai.api_type = "azure"
    openai.api_base = os.getenv("OPENAI_API_BASE")
    openai.api_version = "2023-07-01-preview"
    openai.api_key = os.getenv("OPENAI_API_KEY")

    # Process the message using the Chat API
    completion = openai.ChatCompletion.create(
        engine=os.getenv("OPENAI_API_ENGINE"),
        messages=message_text,
        temperature=0.7,
        max_tokens=800,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None
    )

    return completion