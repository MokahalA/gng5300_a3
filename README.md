# Shopping Assistant Chatbot

This is a skincare products command-line chatbot implementation using LangGraph.

## Folder Structure
- `chatbot.py`: Main entry point of the application which runs the chatbot & implements the graph.
- `setup.py`: This file must be run in order to set up the SQLite database files.
- `tools.py`: Contains the LangChain tools for the chatbot.
- `assistant.py`: Contains the State and Assistant objects.
- `skincare_products.csv`: Raw data for 48 skincare products.

A **detailed report** explaining the design and implementation of the system has been provided in the repository as `A3_Report.pdf`. You may also access the report via the following [Google Docs link](https://docs.google.com/document/d/1phvv-uX34RrG9w8Xt4ZW_MiRagiqWRdcWMDiYbSb778/edit?usp=sharing).


## How to Run

**NOTE:** You must have a locally installed version of Ollama and the llama3.2:3b model to be able to run the chatbot. (16GB of RAM needed)

https://ollama.com/library/llama3.2:3b 

Set up a virtual environment and activate it:
    
    python -m venv venv
    
    venv/Scripts/activate

Install the dependencies:

    pip install -r requirements.txt

Run the setup file to create the database files:

    python setup.py

To run the CLI program, simply execute the following command in your terminal or command prompt:

    python chatbot.py
