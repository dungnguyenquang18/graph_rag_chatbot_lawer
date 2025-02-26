from main_model import Model
from model import GAE
from rapidfuzz import process  # For fuzzy matching
import torch.nn.functional as F
from collections import deque
import torch

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
                    print(formatted_path)
            else:
                print(f"{match_name} -> NO_DIRECT_RELATION -> {similar_node_name} with score {similarity_score:.4f}")
