# Medical Chatbot

## How to Run

To run the Medical Chatbot application, you will need to execute the command `run shiny app.py` in your terminal or command prompt. This will start the application and allow it to receive incoming questions from users.

## How to Deploy

To deploy the Medical Chatbot application onto Shiny website, you can use the following command: `rsconnect deploy <PATH> --name ACCOUNT_NAME --title APP_NAME`. Assumes you have an API key entered. Replace <PATH> with the file path of your application, ACCOUNT_NAME with your Shiny account name, and APP_NAME with the name you want to give your deployed application.

## System Design Diagram

The Medical Chatbot application uses Artificial Intelligence (AI) algorithms to understand users' questions and provide appropriate answers. It leverages OpenAI website to provides the answer to the user's question. Here is how the system design diagram for the Medical Chatbot application looks like:

<p align='center'><img src="https://github.com/yiqiao-yin/medical-chatbot/blob/main/docs/diagram.png"></img></p>

As depicted in the diagram, the Medical Chatbot application makes use of three different components. Firstly, the user interacts with the chatbot interface through a web browser. Secondly, the input provided by the user is sent to the AI algorithm that understands natural language and generates the query to be sent to the third component, which is the OpenAI website. Lastly, OpenAI returns the relevant answer to the initial user query.