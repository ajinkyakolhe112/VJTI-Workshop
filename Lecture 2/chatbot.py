from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def create_chatbot():
    # Initialize the chat model
    llm = ChatOpenAI(
        temperature=0.7,
        model_name="gpt-3.5-turbo"
    )
    
    # Create a conversation chain with memory
    conversation = ConversationChain(
        llm=llm,
        memory=ConversationBufferMemory(),
        verbose=True
    )
    
    return conversation

def main():
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Please set your OpenAI API key in the .env file")
        return
    
    # Create the chatbot
    chatbot = create_chatbot()
    
    print("Welcome to the Chatbot! (Type 'quit' to exit)")
    print("-" * 50)
    
    while True:
        # Get user input
        user_input = input("\nYou: ").strip()
        
        if user_input.lower() == 'quit':
            print("\nGoodbye!")
            break
        
        # Get response from the chatbot
        response = chatbot.predict(input=user_input)
        
        print("\nChatbot:", response)

if __name__ == "__main__":
    main() 