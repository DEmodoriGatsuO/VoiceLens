# Author: De'modori Gatsuo
# Date: 2024-01-30
# Description: This script uses the Azure AI Service to read the text in front of you using OCR, OpenAI proofreading, and Text to Speech
# Updates:
#   2024-01-30: launch


import azure.functions as func
import azure.cognitiveservices.speech as speechsdk
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
from msrest.authentication import CognitiveServicesCredentials
import os
import openai
import time
import base64
import logging
from io import BytesIO

app = func.FunctionApp(http_auth_level=func.AuthLevel.FUNCTION)

@app.route(route="voicelens", methods=['POST'])
def voicelens(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    # message = req.params.get('message')
    encoded_image_data = req.params.get('image')
    if not encoded_image_data:
        try:
            req_body = req.get_json()
        except ValueError:
            pass
        else:
            encoded_image_data = req_body.get('image')
            # 1. Read API - OCR
            image_text = extract_text_from_image(encoded_image_data)
            # 2. OpenAI
            message = get_chat_completion(image_text)
            # 3. Text to Speech
            audio_data = synthesize_speech_to_audio_data(message)

    if image_text:
        return func.HttpResponse(audio_data)
    else:
        return func.HttpResponse(
             "This HTTP triggered function executed successfully.",
             status_code=200
        )

def extract_text_from_image(encoded_image_data):
    """
    Extracts text from the given base64-encoded image data using the Azure Computer Vision API.

    Args:
    encoded_image_data (str): The base64-encoded data of the image from which to extract the text.

    Returns:
    str: Extracted text from the image or None if the extraction fails.
    """
    # Decode the base64-encoded data
    image_data = base64.b64decode(encoded_image_data)    
    # Authenticate
    subscription_key = os.environ["MULTI_SERVICE_KEY"]
    endpoint = os.environ["MULTI_SERVICE_ENDPOINT"]

    computervision_client = ComputerVisionClient(endpoint, CognitiveServicesCredentials(subscription_key))

    # Call API with image data
    try:
        image_stream = BytesIO(image_data)
        read_response = computervision_client.read_in_stream(image_stream,  raw=True)

        # Get the operation location (URL with an ID at the end) from the response
        read_operation_location = read_response.headers["Operation-Location"]
        operation_id = read_operation_location.split("/")[-1]

        # Call the "GET" API and wait for it to retrieve the results 
        while True:
            read_result = computervision_client.get_read_result(operation_id)
            if read_result.status not in ['notStarted', 'running']:
                break
            time.sleep(1)

        # Extract and return text if operation succeeded
        if read_result.status == OperationStatusCodes.succeeded:
            return "".join(
                line.text for text_result in read_result.analyze_result.read_results
                for line in text_result.lines
            )
    except Exception as e:
        logging.info(f"An error occurred: {e}")
        return None

def get_chat_completion(message_text):
    # Configure the OpenAI API settings
    openai.api_type = "azure"
    openai.api_base = os.getenv("OPENAI_API_BASE")
    openai.api_version = "2023-07-01-preview"
    openai.api_key = os.getenv("OPENAI_API_KEY")
    _messages =  [
        {
            "role": "system",
            "content": ("Below is the text read by OCR. The following is an "
                        "introduction to the Japanese text. Please check the "
                        "content and make sure the text is appropriate. If it is "
                        "not correct, please complete it so that it can be read out. "
                        "・Don't change the numerical values in the sentences.")
        },
        {
            "role": "user",
            "content": message_text
        }
    ]
    # Process the message using the Chat API
    completion = openai.ChatCompletion.create(
        engine=os.getenv("OPENAI_API_ENGINE"),
        messages=_messages,
        temperature=0.7,
        max_tokens=500,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None
    )

    return completion['choices'][0]['message']['content']

def synthesize_speech_to_audio_data(text):
    """
    Converts the given text to speech and returns the base64 encoded audio data.

    Args:
    text (str): Text to be converted to speech.

    Returns:
    str: The base64 encoded audio data as a string.
    """

    # Configure Azure Speech Service
    speech_config = speechsdk.SpeechConfig(subscription=os.environ.get('MULTI_SERVICE_KEY'), 
                                           region=os.environ.get('MULTI_SERVICE_REGION'))
    audio_config = speechsdk.audio.AudioOutputConfig(use_default_speaker=True)

    # Set the voice for speech synthesis
    speech_config.speech_synthesis_voice_name='ja-JP-KeitaNeural'

    # Create an instance of SpeechSynthesizer
    speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, 
                                                     audio_config=audio_config)

    # Convert text to speech
    speech_synthesis_result = speech_synthesizer.speak_text_async(text).get()

    # If synthesis is successful, encode and return the audio data
    if speech_synthesis_result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        audio_data = speech_synthesis_result.audio_data
        return base64.b64encode(audio_data).decode('utf-8')

    # If synthesis fails, raise an exception
    elif speech_synthesis_result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = speech_synthesis_result.cancellation_details
        raise Exception(f"Speech synthesis canceled: {cancellation_details.reason}, "
                        f"Error details: {cancellation_details.error_details}")