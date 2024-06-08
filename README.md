# Toward Repeatable, Understandable and Collaborative: Modeling, Storing and Tracing Full Lifecycle of Machine Learning

**Due to the double-blind principle, we do not provide any author information or contact details until the paper review process is completed.**

This paper has been submitted to *SSDBM 2024*.

<img src="https://github.com/MLFlowDB/MLFlowDB/blob/main/README.assets/Logo.png?raw=true" alt="Logo" style="zoom:50%;" />

##  Abstract

Machine learning (ML) has emerged as a pivotal research methodology across numerous domains. The ML lifecycle is heterogeneous, comprising various stages, each involving distinct domain knowledge, data, operations, and outputs, all of which are interrelated. Thus, a holistic understanding of the ML lifecycle is vital for both sharing and debugging ML models. Achieving such an understanding necessitates the storage and management of data, code, models, and their interrelationships throughout the ML lifecycle. To overcome these challenges, we introduce MLDR, a new methodology for representing the ML lifecycle, focusing on early data stages, and balancing scalability, reproducibility, and flexibility. Built on MLDR, we developed MLFlowDB, a comprehensive ML lifecycle management platform. MLFlowDB stores various lifecycle elements and their interrelations and enables detailed trace analysis across upstream, downstream, and historical dimensions. Furthermore, it enables the export and sharing of trace outcomes. The effectiveness of MLDR in supporting ML reproducibility was assessed and compared against three other ML lifecycle representation methods, highlighting MLDR's advantages in lifecycle coverage and semantic representation. Moreover, the successful validation of different use cases of analysis in real-world ML workflows demonstrates the effectiveness of MLFlowDB in enhancing the analysis of the ML lifecycle.

## Directory Structure:

-   **MLDR**: A storage directory for content related to MLDR.
-   **MLFlowDB**: A storage directory for content related to MLFlowDB.
