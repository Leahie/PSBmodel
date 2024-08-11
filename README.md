# PSBmodel

## Overview 

Repository for **Charting the Evolution and Transformative Impact of the Pacific Symposium on Biocomputing Through a 30-Year Retrospective Analysis of Collaborative Networks and Themes Using Modern Computational Tools.**. 

Founded nearly 30 years ago, the Pacific Symposium on Biocomputing (PSB) has continually promoted collaborative research in computational biology, annually highlighting emergent themes that reflect the expanding interdisciplinary nature of the field. This study aimed to explore the collaborative and thematic dynamics at PSB using topic modeling and network analysis methods. We we identified 14 central topics that have characterized the discourse at PSB over the past three decades. Our findings demonstrate significant trends in topic relevance, with a growing emphasis on machine learning and integrative analyses. We observed not only an expanding nexus of collaboration but also PSB’s crucial role in fostering interdisciplinary collaborations. It remains unclear, however, whether the shift towards interdisciplinarity was driven by the conference itself, external academic trends, or broader societal shifts towards integrated research approaches. Future applications of next-generation analytical methods may offer deeper insights into these dynamics. Additionally, we have developed a web application that leverages retrieval augmented generation and large language models, enabling users to efficiently explore past PSB proceedings.

## Project Setup 
The code is split into two parts: PSB Topic Modeling and PSB Collaboration Networks. The Topic Modeling code, '/PSB_Topic_Modeling' includes how we web scrapped the data and used LDA and Bertopic to model the changes in topics over time. The Collaboration Network code, '/PSB_Network', includes the code for obtaining the citations and modeling the network. 

If you would like to use our training data, it is in this [drive](https://drive.google.com/drive/folders/1uvPSGsPaSboP7TnobM3m5kLG1hoTNZc6?usp=sharing). We used the 'Main Bodies' section to train our model, but you can experiement and try out the model with full text if curious. 

## Webapplication
In addition to the main code, [here](https://psb-rag.streamlit.app/) is a link to a web application that enables users to efficiently explore past proceedings. /
Below is a video tutorial of how to use it /
[![Video Tutorial](https://img.youtube.com/vi/t4ghzrL7AgM/0.jpg)](https://www.youtube.com/watch?v=t4ghzrL7AgM)
