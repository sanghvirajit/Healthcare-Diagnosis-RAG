# Healthcare Diagnosis with RAG
Enhancing Healthcare Diagnostics with Retrieval-Augmented Generation (RAG): Leveraging LangChain, ChromaDB, and Hugging Face

![image](https://github.com/user-attachments/assets/a2a1d92b-aba5-443d-b672-cdba8fcc56c7)


# Introduction
The integration of AI in healthcare is revolutionising diagnostics, particularly with advancements in Retrieval-Augmented Generation (RAG). By combining retrieval-based knowledge with generative capabilities, RAG enhances traditional AI models, delivering more accurate and reliable results. In this blog, we explore how RAG can be applied to diagnosing Parkinson's disease using gait analysis. Leveraging LangChain, ChromaDB vector database for efficient data retrieval and LLaMA-3 via the Groq API for intelligent generation, we demonstrate a cutting-edge approach to AI-driven medical diagnostics.

LangChain is a popular open-source framework designed for developing applications powered by Large Language Models (LLMs). It provides tools and abstractions to integrate LLMs with external data sources, memory, and reasoning capabilities, making it easier to build AI-driven applications such as chatbots, RAG (Retrieval-Augmented Generation) systems, and autonomous agents.


![image](https://github.com/user-attachments/assets/32c8d505-9564-4229-a73b-384745dbdf1b)


## Clone the github repository
```sh
git clone https://github.com/sanghvirajit/Healthcare_diagnosis_RAG.git
```

## Create a virtual environment
```sh
conda create --name rag_env python=3.11
```

## Activate the virtual environment
```sh
conda activate rag_env
```

## Install the required libraries
```sh
pip install -r requirements.txt
```

## Run the local Flask API
```sh
uvicorn rag:app --host 0.0.0.0 --port 8000
```

## Send the request and get the response
```sh
python rag_response.py
```

## Knowledge Base PDF

[Link Text] https://www.frontiersin.org/journals/neurology/articles/10.3389/fneur.2024.1472956/full

## Gait Analysis Data

![image](https://github.com/user-attachments/assets/9ca07b96-f969-4af3-a251-8614084825dc)

```sh
{
    "result_details": [
        {
            "name": "Gait Symmetry",
            "description": "The gait symmetry indicates how symmetrical the left and right step length is. The higher the value, the better.",
            "score_dict": {
                "nominal_value": 80,
                "nominal_lower": 90,
                "nominal_upper": 100,
                "unit": "%"
            }
        },
        {
            "name": "Gait Speed",
            "description": "The gait speed indicates how fast the patient was walking during the analysis.",
            "score_dict": {
                "nominal_value": 1.5,
                "nominal_lower": 2,
                "nominal_upper": 5,
                "unit": "km/h"
            }

        },
        {
            "name": "Cadence",
            "description": "The gait cadence indicates the total number of steps taken within a minute during the analysis.",
            "score_dict": {
                "nominal_value": 80,
                "nominal_lower": 90,
                "nominal_upper": 115,
                "unit": "Steps/min"
            }
        },
        {
            "name": "Double Support Left",
            "description": "The gait double support left indicates proportion of time that both feet of the patient are on the ground during the gait cycle of left leg.",
            "score_dict": {
                "nominal_value": 30,
                "nominal_lower": 10,
                "nominal_upper": 40,
                "unit": "%"
            }
        },
        {
            "name": "Double Support Right",
            "description": "The gait double support right indicates proportion of time that both feet of the patient are on the ground during the gait cycle of right leg.",
            "score_dict": {
                "nominal_value": 35,
                "nominal_lower": 10,
                "nominal_upper": 40,
                "unit": "%"
            }
        },
        {
            "name": "Gait Variability Left",
            "description": "The gait variability indicates step-to-step length fluctuation of left / right leg during the analysis.",
            "score_dict": {
                "nominal_value": 7,
                "nominal_lower": 0,
                "nominal_upper": 5,
                "unit": "%"
            }
        },
        {
            "name": "Gait Variability Right",
            "description": "The gait variability indicates step-to-step length fluctuation of left / right leg during the analysis.",
            "score_dict": {
                "nominal_value": 8,
                "nominal_lower": 0,
                "nominal_upper": 5,
                "unit": "%"
            }
        },
        {
            "name": "Step Length Left",
            "description": "The left gait step length indicates the average distance between the point of initial contact of the left foot to the point of initial contact of the right foot during the analysis.",
            "score_dict": {
                "nominal_value": 50,
                "nominal_lower": 55,
                "nominal_upper": 80,
                "unit": "cm"
            }
        },
        {
            "name": "Step Length Right",
            "description": "The right gait step length indicates the average distance between the point of initial contact of the right foot to the point of initial contact of the left foot during the analysis.",
            "score_dict": {
                "nominal_value": 48,
                "nominal_lower": 55,
                "nominal_upper": 80,
                "unit": "cm"
            }
        },
        {
            "name": "Step Time Left",
            "description": "The gait step time indicates the average time elapsed from initial contact of the left foot to initial contact of the right foot during the analysis.",
            "score_dict": {
                "nominal_value": 0.59,
                "nominal_lower": 0.51,
                "nominal_upper": 0.65,
                "unit": "s"
            }
        },
        {
            "name": "Step Time Right",
            "description": "The gait step time indicates the average time elapsed from initial contact of the right foot to initial contact of the left foot during the analysis.",
            "score_dict": {
                "nominal_value": 0.61,
                "nominal_lower": 0.51,
                "nominal_upper": 0.65,
                "unit": "s"
            }
        },
        {
            "name": "Stance Time Left",
            "description": "The gait stance time indicates the average percentage of time during which the left foot is in contact with the ground during the analysis.",
            "score_dict": {
                "nominal_value": 80,
                "nominal_lower": 55,
                "nominal_upper": 65,
                "unit": "%"
            }
        },
        {
            "name": "Stance Time Right",
            "description": "The gait stance time indicates the average percentage of time during which the right foot is in contact with the ground during the analysis.",
            "score_dict": {
                "nominal_value": 75,
                "nominal_lower": 55,
                "nominal_upper": 65,
                "unit": "%"
            }
        }
    ]
}
```

## Query

Following query would be generated automatically from the input gait analysis results in JSON format saved under ```data``` dir.

```
Patient gait analysis results: Gait Symmetry for the patient is 80 % and 
the reference range for healthy patient is between 90 % and 100 %, 
Gait Speed for the patient is 1.5 km/h and the reference range for healthy 
patient is between 2 km/h and 5 km/h, Cadence for the patient is 80 Steps/min 
and the reference range for healthy patient is between 90 Steps/min and 
115 Steps/min, Double Support Left for the patient is 30 % and the 
reference range for healthy patient is between 10 % and 40 %, 
Double Support Right for the patient is 35 % and the reference range for 
healthy patient is between 10 % and 40 %, 
Gait Variability Left for the patient is 7 % 
and the reference range for healthy patient is between 0 % and 5 %, 
Gait Variability Right for the patient is 8 % and the reference range for 
healthy patient is between 0 % and 5 %, 
Step Length Left for the patient is 50 cm and the reference range for 
healthy patient is between 55 cm and 80 cm, 
Step Length Right for the patient is 48 cm and 
the reference range for healthy patient is between 55 cm and 80 cm, 
Step Time Left for the patient is 0.59 s and the reference range for 
healthy patient is between 0.51 s and 0.65 s, 
Step Time Right for the patient is 0.61 s and the 
reference range for healthy patient is between 0.51 s and 0.65 s, 
Stance Time Left for the patient is 80 % and the reference range for 
healthy patient is between 55 % and 65 %, 
Stance Time Right for the patient is 75 % and the reference range for 
healthy patient is between 55 % and 65 %, 
What does this indicate about Parkinson's disease?
```

## Diagnosis Result

Following diagnosis would be generated based on the query and the knowledge base pdf saved under ``` knowledge_base ``` dir

```
The patient's gait analysis results indicate several abnormalities that are 
relevant to Parkinson's disease. The patient's Gait Speed of 1.5 km/h is 
significantly lower than the reference range for healthy patients, which 
is between 2 km/h and 5 km/h. This suggests a slower gait speed, which is a 
common feature in Parkinson's disease patients.  The patient's Cadence of 
80 Steps/min is also lower than the reference range for healthy patients, 
which is between 90 Steps/min and 115 Steps/min. This indicates a reduced 
walking speed, which is consistent with Parkinson's disease.  The patient's 
Double Support Left and Double Support Right are both higher than the 
reference range for healthy patients, indicating a longer time spent with 
both feet on the ground. This is a characteristic feature of Parkinson's 
disease, where patients often experience difficulty with gait initiation 
and termination.  The patient's Step Length Left and Step Length Right are 
both shorter than the reference range for healthy patients, which is between 
55 cm and 80 cm. This suggests a reduced stride length, which is a common 
feature in Parkinson's disease patients.  The patient's Gait Variability Left 
and Gait Variability Right are both higher than the reference range for 
healthy patients, indicating increased variability in gait patterns. 
This is consistent with Parkinson's disease, where patients often experience 
difficulty with gait coordination and stability.  
The patient's Stance Time Left and Stance Time Right are both higher than the 
reference range for healthy patients, indicating a longer time spent in the 
stance phase of gait. This is a characteristic feature of Parkinson's disease, 
where patients often experience difficulty with gait initiation and 
termination.  Considering these abnormalities, the likelihood of Parkinson's 
disease is moderate. However, possible further assessment with the doctors is 
recommended and diagnosis should be confirmed by a qualified medical 
professional.
```




