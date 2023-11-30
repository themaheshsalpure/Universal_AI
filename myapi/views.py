from django.http import HttpResponse,JsonResponse
from . import NeuralNetwork
from . import Brain
import random
import json
import torch
import random

from NeuralNetwork import bag_of_words,tokenize
import random
import json
from Brain import NeuralNet

# ---------------
# ----------------------------------------------------------------------------------------

from huggingface_hub import login
login(token="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")

import argparse
# import parser
parser = argparse.ArgumentParser()
parser.add_argument('--type', default='torch.FloatTensor', help='type of tensor - e.g torch.HalfTensor')
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
# model = AutoModelForCausalLM.from_pretrained("pansophic/rocket-3B", trust_remote_code=True, torch_dtype=torch.bfloat16)

# ---------------------------------------------------------------------------------- IMPORTED FROM JARVIS
# device = torch.device('cuda' if torch .cuda.is_available() else 'cpu')


model = AutoModelForCausalLM.from_pretrained("pansophic/rocket-3B", trust_remote_code=True, torch_dtype=torch.bfloat16).to("cuda")
tokenizer = AutoTokenizer.from_pretrained("pansophic/rocket-3B", trust_remote_code=True, torch_dtype=torch.bfloat16)
streamer = TextStreamer(tokenizer)




# Create your views here.
def home(request):
    return HttpResponse("API")


def getResponse(request, user_message):
    user_message = str(user_message).lower()
    prompt = """<|im_start|>system
    {system}<|im_end|>
    <|im_start|>user
    {user}<|im_end|>
    <|im_start|>assistant
    """

    # system = "You are a wellness guru who transforms every health remedy into a time-honored secret, imparting ancient wisdom and traditional practice. Provide insight and guidance on traditional Indian cure for common ailment with a touch of holistic wisdom"
    system = "As a revered wellness guru, you unveil the ancient secrets of traditional Indian cures for common ailments, offering profound insights and guidance. Share your wisdom one remedy at a time, infusing each response with the time-honored essence of holistic health practices"
    # user = "I am having joint pain what should i do now"
    user = user_message

    # Apply the ChatML format
    prompt = prompt.format(system=system, user=user)

    # Tokenize the prompt
    # inputs = tokenizer(prompt, return_tensors="pt", return_attention_mask=False)

    # CHNAGED FROM GPU TO CPU

    inputs = tokenizer(prompt, return_tensors="pt", return_attention_mask=False).to("cuda")
    generated_text = model.generate(**inputs, max_length=200, top_p=0.95, do_sample=True, temperature=0.7, use_cache=True, streamer=streamer)

    generated_text_ids = generated_text[0].tolist()
    reply = tokenizer.decode(generated_text_ids, skip_special_tokens=True)

    model_response = {'response': reply}
    return JsonResponse(model_response)


def makeAppointment(request, user_message):
    user_message = user_message.lower()
    if user_message.count("book") >=1 or user_message.count("appointment") >=1:
        user_message = "Select near by hospitals to book an appointment like Shantanu Hospital, rahul Hospital, sanket Hospital or shreyash hospital"
    elif user_message.count("shantanu") >=1 :
        user_message = "7083694063"
    elif user_message.count("Mahesh") >=1 :
        user_message = "9403512671"
    elif user_message.count("sanket") >=1 :
        user_message = "7083694063"
    elif user_message.count("shreyash") >=1 or user_message.count("shreyas") >=1 :
        user_message = "9403512671"
    else:
        user_message = "Please choose a hospital first."
    

    model_response = {'response': user_message}
    return JsonResponse(model_response)