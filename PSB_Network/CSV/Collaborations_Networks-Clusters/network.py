import pandas as pd
import json
import re
import networkx as nx
import os
import numpy as np
import igraph as ig
import leidenalg
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

def calculate_average_topic_distribution(author_papers):
    if not author_papers:
        return np.zeros(len(author_papers[0]))
    return np.mean(author_papers, axis=0)

def parse_string_to_list(s, num_topics=14):
    try:
#         print(s)
        return np.fromstring(s.strip('[]'), sep=' ')
    except:
        return np.zeros(num_topics)
    
file_path = 'Full_Author_Topic_w_2002.csv'
output_folder = 'coauthorship_networks'

def create_cumulative_coauthorship_networks(file_path, output_folder):
    # Load the CSV file
    df = pd.read_csv(file_path)

    # Initialize a dictionary to hold cumulative graphs for each year
    cumulative_graphs = {}
    author_topic_distributions = {}

    # Create the output folder if it does not exist
    os.makedirs(output_folder, exist_ok=True)

    # Create coauthorship networks
    for year in sorted(df['Year'].unique()):
        print(year)

        yearly_data = df[df['Year'] <= year]
        
        G = nx.Graph()
        
        for _, row in yearly_data.iterrows():
            authors_dict = eval(row['Full Authors'])
            topic_distribution = parse_string_to_list(row['distr'])
            for author_id, author_name in authors_dict.items():
                if author_id not in author_topic_distributions:
                    author_topic_distributions[author_id] = []
                author_topic_distributions[author_id].append(topic_distribution)
                if author_id not in G:
                    G.add_node(author_id, name=author_name)

            author_ids = list(authors_dict.keys())
            for i, author_id in enumerate(author_ids):
                for coauthor_id in author_ids[i+1:]:
                    if G.has_edge(author_id, coauthor_id):
                        G[author_id][coauthor_id]['weight'] += 1
                    else:
                        G.add_edge(author_id, coauthor_id, weight=1)

        # Calculate average topic distributions for authors
        avg_topic_distributions = {author_id: calculate_average_topic_distribution(papers)
                                   for author_id, papers in author_topic_distributions.items()}

        # Store average topic distributions as node attributes
        for author_id, avg_distribution in avg_topic_distributions.items():
            if author_id in G:
                G.nodes[author_id]['avg_topic_distribution'] = avg_distribution.tolist()

        # Calculate author alignment (topic similarity) for each edge
        for u, v, data in G.edges(data=True):
            if u in avg_topic_distributions and v in avg_topic_distributions:
                similarity = cosine_similarity([avg_topic_distributions[u]], [avg_topic_distributions[v]])[0][0]
                G[u][v]['alignment'] = similarity

        cumulative_graphs[year] = G
        
    return cumulative_graphs

cumul_graphs = create_cumulative_coauthorship_networks(file_path, output_folder)

for year, graph in cumul_graphs.items():
    nx.write_gml(graph, os.path.join(output_folder, f"cumulative_coauthorship_network_{year}.gml"))

for year, graph in cumul_graphs.items():
    ig_graph = ig.Graph.from_networkx(graph)

    # Run Leiden clustering
    partition = leidenalg.find_partition(ig_graph, leidenalg.ModularityVertexPartition)

    # Save clustering results
    clusters = {vertex['name']: cluster for vertex, cluster in zip(ig_graph.vs, partition.membership)}
    with open(os.path.join(output_folder, f"leiden_clustering_cumulative_{year}.json"), 'w') as f:
        json.dump(clusters, f)
    
    print(year)

def create_cs_coauthorship_networks(file_path, output_folder):
    # Load the CSV file
    df = pd.read_csv(file_path)
    
    cross_sectional_graphs = {}

    # Create the output folder if it does not exist
    os.makedirs(output_folder, exist_ok=True)

    # Create coauthorship networks
    for year in sorted(df['Year'].unique()):
        print(year)
        yearly_data = df[df['Year'] == year]
        
        author_topic_distributions = {}
        G = nx.Graph()
        
        for _, row in yearly_data.iterrows():
            authors_dict = eval(row['Full Authors'])
            topic_distribution = parse_string_to_list(row['distr'])
            for author_id, author_name in authors_dict.items():
                if author_id not in author_topic_distributions:
                    author_topic_distributions[author_id] = []
                author_topic_distributions[author_id].append(topic_distribution)
                if author_id not in G:
                    G.add_node(author_id, name=author_name)

            author_ids = list(authors_dict.keys())
            for i, author_id in enumerate(author_ids):
                for coauthor_id in author_ids[i+1:]:
                    G.add_edge(author_id, coauthor_id)

        # Calculate average topic distributions for authors
        avg_topic_distributions = {author_id: calculate_average_topic_distribution(papers)
                                   for author_id, papers in author_topic_distributions.items()}

        # Store average topic distributions as node attributes
        for author_id, avg_distribution in avg_topic_distributions.items():
            if author_id in G:
                G.nodes[author_id]['avg_topic_distribution'] = avg_distribution.tolist()

        # Calculate author alignment (topic similarity) for each edge
        for u, v, data in G.edges(data=True):
            if u in avg_topic_distributions and v in avg_topic_distributions:
                similarity = cosine_similarity([avg_topic_distributions[u]], [avg_topic_distributions[v]])[0][0]
                G[u][v]['alignment'] = similarity

        cross_sectional_graphs[year] = G
        
    return cross_sectional_graphs

cs_graphs = create_cs_coauthorship_networks(file_path, output_folder)

print(cs_graphs)

for year, graph in cs_graphs.items():
    nx.write_gml(graph, os.path.join(output_folder, f"cross_sectional_coauthorship_network_{year}.gml"))

for year, graph in cs_graphs.items():
        ig_graph = ig.Graph.from_networkx(graph)
        
        # Run Leiden clustering
        partition = leidenalg.find_partition(ig_graph, leidenalg.ModularityVertexPartition)

        # Save clustering results
        clusters = {vertex['name']: cluster for vertex, cluster in zip(ig_graph.vs, partition.membership)}
        with open(os.path.join(output_folder, f"leiden_clustering_cross_sect_{year}.json"), 'w') as f:
            json.dump(clusters, f)
        
        print(year)