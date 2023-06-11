import os
from threading import Thread
import gradio as gr
from queue import SimpleQueue
from langchain import PromptTemplate, OpenAI, LLMChain
from callbacks import StreamingGradioCallbackHandler, job_done

os.environ['OPENAI_API_KEY'] = ""

q = SimpleQueue()
llm = OpenAI(temperature=0, streaming=True, callbacks=[StreamingGradioCallbackHandler(q)])

prompt_template = '''You are an intelligent AI chatbot having a conversation with a human. You are given context, and a user request. Try to satisfy the user request using the context if it is helpful.

context:
{context}

Human: {request}
AI:'''
prompt = PromptTemplate(template=prompt_template, input_variables=["request", "context"])

llm_chain = LLMChain(
    llm=llm,
    prompt=prompt
)

def add_text(history, text):
    history = history + [(text, None)]
    return history, ""

def streaming_chat(history, key):
    user_input = history[-1][0]
    thread = Thread(target=llm_chain.predict, kwargs={"request": user_input, "context": history})
    thread.start()
    history[-1][1] = ""
    while True:
        next_token = q.get(block=True) # Blocks until an input is available
        if next_token is job_done:
            break
        history[-1][1] += next_token
        yield history
    thread.join()


with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    textbox = gr.Textbox("Hello, how are you doing today?")

    textbox.submit(add_text, [chatbot, textbox], [chatbot, textbox]).then(
        streaming_chat, chatbot, chatbot
    )

demo.queue().launch(server_port=1235)
    