from neo4j import GraphDatabase
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
import random
from rapidfuzz import process  # For fuzzy matching
import torch.nn.functional as F
from collections import deque
from model import GAE
from main_model import Model


driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "vinh1950"))
def get_graph_data():
    with driver.session() as session:
        # Get nodes
        nodes_query = "MATCH (n:Entity) RETURN id(n) AS node_id, n.name AS name"
        nodes = session.run(nodes_query)
        node_mapping = {record["node_id"]: record["name"] for record in nodes}

        # Get edges and edge types
        edges_query = "MATCH (n)-[r]->(m) RETURN id(n) AS source, id(m) AS target, type(r) AS relationship_type"
        edges = session.run(edges_query)
        edge_list = [(record["source"], record["target"], record["relationship_type"]) for record in edges]

        # Extract all unique relationship types
        unique_relationship_types = {e[2] for e in edge_list}

        return node_mapping, edge_list, unique_relationship_types



def loss_function(reconstructed, edge_index):
        # Binary cross-entropy loss for adjacency reconstruction
        # Một tensor toàn giá trị 1 với chiều dài bằng số lượng edges (M) trong đồ thị.
        target = torch.ones(edge_index.size(1))  # All edges exist
        # Sự khác biệt giữa logits dự đoán (reconstructed) và các giá trị thực (target).
        return F.binary_cross_entropy_with_logits(reconstructed, target)



def find_indirect_connection(edge_list, node_mapping, start, target, max_depth=10):
    """
    Find indirect connections between two nodes using BFS.

    Parameters:
        edge_list (list): List of edges as (source, target, relationship).
        node_mapping (dict): Mapping of node IDs to node names.
        start (int): The starting node ID.
        target (int): The target node ID.
        max_depth (int): Maximum depth to explore for indirect connections.

    Returns:
        list: A list of paths from start to target.
    """
    # Build graph as adjacency list: {node: [(neighbor, relationship)]}
    graph = {}
    for src, tgt, rel in edge_list:
        if src not in graph:
            graph[src] = []
        if tgt not in graph:
            graph[tgt] = []
        graph[src].append((tgt, rel))
        graph[tgt].append((src, rel))  # Assuming undirected graph for traversal

    # Initialize BFS
    queue = deque([(start, [], 0)])  # (current_node, path_so_far, current_depth)
    visited = set()

    paths = []

    while queue:
        current_node, path, depth = queue.popleft()

        if depth > max_depth:  # Stop exploring if max depth is exceeded
            continue

        if current_node == target:  # Target node found
            paths.append(path)
            continue

        # Mark node as visited
        visited.add(current_node)

        # Explore neighbors
        for neighbor, relationship in graph.get(current_node, []):
            if neighbor not in visited:
                queue.append((neighbor, path + [(current_node, relationship, neighbor)], depth + 1))

    return paths

def find_closest_entities(entities, node_mapping):
    """
    Finds the closest matching entities in node_mapping for a list of query entities.

    Parameters:
        entities (list): List of entity names to match.
        node_mapping (dict): Mapping of node IDs to entity names.

    Returns:
        list: A list of tuples [(query_entity, closest_match_id, closest_match_name, score)].
    """
    results = []
    node_names = list(node_mapping.values())
    for entity in entities:
        closest_match, score, index = process.extractOne(entity, node_names)
        closest_match_id = list(node_mapping.keys())[index]
        results.append((entity, closest_match_id, closest_match, score))
    return results




if __name__ == '__main__':
    node_mapping, edge_list, unique_relationship_types = get_graph_data()
    # Create a mapping for edge names to indices
    edge_name_to_index = {name: idx for idx, name in enumerate(set(edge[2] for edge in edge_list))}

    # Convert edge list to indices
    edge_index = [(src, tgt, edge_name_to_index[name]) for src, tgt, name in edge_list]
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    reverse_node_mapping = {value: key for key, value in node_mapping.items()}
        
        
    # Step 1: Create edge_index
    edge_index = torch.tensor([[e[0], e[1]] for e in edge_list], dtype=torch.long).t().contiguous()

    # Features for nodes (one-hot encoding)
    num_nodes = len(node_mapping)
    features = torch.eye(num_nodes)  # One-hot encoding for each node

    # Split edges into training and validation sets
    num_edges = edge_index.size(1)
    indices = list(range(num_edges))
    random.shuffle(indices)
    split_idx = int(0.8 * num_edges)  # 80% for training, 20% for validation

    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]

    train_edge_index = edge_index[:, train_indices]
    val_edge_index = edge_index[:, val_indices]

    # Step 2: Create Data objects for training and validation
    train_data = Data(x=features, edge_index=train_edge_index)
    val_data = Data(x=features, edge_index=val_edge_index)

    target = torch.ones(edge_index.size(1))
    
    # Step 4: Initialize the model
    input_dim = features.size(1)
    hidden_dim = 16
    embedding_dim = 8
    model = GAE(input_dim, hidden_dim, embedding_dim)

    # Step 5: Define optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    try:
        checkpoint = torch.load("gae_model.pth")
        model.load_state_dict(checkpoint['model_state_dict'])


        
    except:
        # Step 6: Train the model and validate after each epoch
        model.train()
        for epoch in range(200):
            # Training phase
            optimizer.zero_grad()
            embeddings, reconstructed = model(train_data.x, train_data.edge_index)
            train_loss = loss_function(reconstructed, train_data.edge_index)
            train_loss.backward()
            optimizer.step()

            # Validation phase
            model.eval()
            with torch.no_grad():
                val_embeddings, val_reconstructed = model(val_data.x, val_data.edge_index)
                val_loss = loss_function(val_reconstructed, val_data.edge_index)

            # Switch back to train mode for next epoch
            model.train()

            # Print losses
            print(f'Epoch {epoch + 1}, Train Loss: {train_loss.item()}, Validation Loss: {val_loss.item()}')

        # Lưu trạng thái mô hình
        torch.save(model.state_dict(), "gae_model.pth")
        print("Mô hình đã được lưu!")
    
    llm = Model()
    query = "hãy nêu điều 2 khoản 1 luật đất đai"
    entities, _ = llm.process_llm_out(query)
        
        
    # Input: List of entities extracted from the query
    query_entities = entities  # Replace with your entities

    # Step 1: Find the closest matching nodes for all query entities
    matches = find_closest_entities(query_entities, node_mapping)

    print("Closest matches for query entities:")
    for query_entity, match_id, match_name, score in matches:
        print(f"Query: '{query_entity}' -> Match: '{match_name}' (Node ID: {match_id}) with score {score:.2f}")

    # Step 2: Use embeddings to find similar nodes for each matched entity
    model.eval()
    with torch.no_grad():
        embeddings, _ = model(train_data.x, train_data.edge_index)

    #information
    information = ''
    
    # Normalize node embeddings
    node_embeddings_norm = F.normalize(embeddings, p=2, dim=1)

    # Step 3: Aggregate and analyze results for each matched entity
    K = 20  # Number of top similar nodes to retrieve

    for query_entity, match_id, match_name, score in matches:
        print(f"\nTop-{K} similar nodes for '{query_entity}' (Matched Node: {match_name}):")
        query_embedding = embeddings[match_id]
        query_embedding = F.normalize(query_embedding, p=2, dim=0)

        # Compute similarity scores
        similarity_scores = torch.matmul(query_embedding.unsqueeze(0), node_embeddings_norm.T).squeeze()

        # Retrieve Top-K similar nodes
        top_k_indices = torch.topk(similarity_scores, K).indices

        for idx in top_k_indices:
            similar_node_id = idx.item()
            similarity_score = similarity_scores[idx].item()
            similar_node_name = node_mapping[similar_node_id]

            # Check for direct connection in the edge list
            direct_connections = [
                e for e in edge_list if (e[0] == match_id and e[1] == similar_node_id) or
                                        (e[1] == match_id and e[0] == similar_node_id)
            ]

            # Print details in the desired format
            if direct_connections:
                for connection in direct_connections:
                    source = node_mapping[connection[0]]
                    target = node_mapping[connection[1]]
                    relationship = connection[2]
                    information += f"{source} -> {relationship} -> {target}.\n"
                    print(f"{source} -> {relationship} -> {target} with score {similarity_score:.4f}")
            else:
                # Print indirect connection paths
                paths = find_indirect_connection(edge_list, node_mapping, match_id, similar_node_id)
                if paths:
                    print(f"Indirect paths between '{match_name}' and '{similar_node_name}':")
                    for path in paths:
                        formatted_path = " -> ".join(
                            f"{node_mapping[src]} -[{rel}]-> {node_mapping[tgt]}" for src, rel, tgt in path
                        )
                        information += formatted_path + '.\n'
                        print(formatted_path)
                else:
                    information += f"{match_name} -> NO_DIRECT_RELATION -> {similar_node_name}.\n"
                    print(f"{match_name} -> NO_DIRECT_RELATION -> {similar_node_name} with score {similarity_score:.4f}")

    prompt = f"hãy trả lời {query} từ các thông tin sau đây:\n{information}"
    response = llm.answer(prompt)