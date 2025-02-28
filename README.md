# Graph RAG CHATBOT Lawer

## Overview

This repository contains a Graph-based Retrieval-Augmented Generation (RAG) system designed for a film chatbot. The system utilizes a Neo4j graph database to store and retrieve information about laws and regulations, leveraging machine learning models for entity extraction and relationship mapping.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Directory Structure](#directory-structure)
- [Components](#components)
- [License](#license)

## Installation

1. Clone the repository:

   ```bash
   git clone <repository-url>
   cd graph_rag_for_review_film_chatbot
   ```

2. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

3. Set up your environment variables in a `.env` file:
   ```plaintext
   NEO4J_URI=<your_neo4j_uri>
   NEO4J_USERNAME=<your_neo4j_username>
   NEO4J_PASSWORD=<your_neo4j_password>
   NEO4J_DATABASE=<your_neo4j_database>
   API_KEY=<your_api_key>
   URI=<your_mongodb_uri>
   DB_NAME=<your_database_name>
   COLLECTION_NAME=<your_collection_name>
   DIRECTORY_PATH=<your_directory_path>
   ```

## Usage

1. **Insert Data**: First, run the script to insert data into the Neo4j database.

   ```bash
   python make_data/insert_graph_db.py
   ```

2. **Train the Model**: After the data has been inserted successfully, you can train the model.

   ```bash
   python src/gae/train.py
   ```

3. **Run the Chatbot**: To start the chatbot server, execute:

   ```bash
   python src/serve.py
   ```

## Directory Structure
