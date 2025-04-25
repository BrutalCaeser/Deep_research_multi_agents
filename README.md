# Kairon Deep Research Agent

This project implements a multi-agent system for generating, critiquing, and refining research content using advanced AI models. The system leverages tools like GROQ, OpenAI, and Tavily to create a collaborative workflow for research and content generation.

## Features

- **Planning**: Generates a high-level outline for a research topic.
- **Research**: Uses Tavily to gather relevant information for the task.
- **Content Generation**: Drafts detailed answers based on the outline and research.
- **Reflection**: Critiques the generated content and provides recommendations for improvement.
- **Iterative Refinement**: Revises content based on critique and additional research.

## Project Structure
- **`.env`**: Contains API keys for GROQ, OpenAI, and Tavily. To be provided by the user.
- **`changes.txt`**: Notes about recent changes in the project.
- **`main.py`**: Core implementation of the multi-agent system.
- **`requirements.txt`**: Python dependencies required for the project.

## Setup

1. Clone the repository:
   ```sh
   git clone <repository-url>
   cd <repository-folder>
2. Install Dependencies:
   pip install -r requirements.txt

3. Run the main script
   python main.py


Usage
The system is designed to process tasks iteratively. You can define a task and specify the maximum number of revisions. The agents will collaborate to generate, critique, and refine the content until the desired quality is achieved.

Notes
The project uses GROQ as the default model for content generation. You can switch to OpenAI by uncommenting the relevant lines in main.py.
Tavily is used for research purposes, fetching relevant information based on generated queries.
Contributing
Feel free to fork the repository and submit pull requests for improvements or new features.
