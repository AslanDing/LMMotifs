Dear Reviewer cCc8,

Thank you very much for your time and valuable feedback on our manuscript. We sincerely appreciate your thoughtful comments and suggestions, which have greatly contributed to improving the quality of our paper. Here is our reply.

> 1. Baselines: The baseline should include more detection methods, like Ghostbuster, Raidar, it’s better if GPTZero can be compared.
   
Thanks for providing these baselines; we added Raidar and Biscope to compare. The experiments results are provided on XXX. The Ghostbuster is based on the OpenAI model, ada and davinci-002, however, these models are not available. From the results, LM$^2$otifs still achieve a comparable performance.

> 2. Results: The AUROC score is almost perfect on three datasets. Please double-check the final results.
   
Thansk for the reminder. In this paper, the performance are based on AUROC, which is usually higher than Accuracy. We checked our results and it is good. To address the concern, we also provide the accuracy based results on XXX. From the results, LM$^2$otifs still achieve the comparable performance.

> 3. Motifs evaluation: Understanding this part is difficult. Authors mention 0.1% edges maintain, but there are also 0.05%, 0.5%, 1%, please revise this part. Besides, as shown in the text, 0.94 AUROC on HC3 and M4 cannot be found, and 0.98 on RAID? Please highlight. 

Thanks for pointing this out. In section 6.3, on line 379, we mentioned a case of preserving 0.1% edges. The results of preserving 0.05%, 0.5%, 1% are provided in Figure 3. We will update these descriptions in the next version.

> 4. Robust evaluation: if replace the characteristic patterns, how would the AUROC score keep a high level? How is the noise introduced? 

Thanks for the insigtful questions. We use the explanation as the guided information to generate the adversarial samples. We first extract the explanation motifs from HGTs and MGTs and replace the explanation motifs in MGTs with motifs in HGTs and vice versa. In this paper, we do not use the grident based noise to attack. Instead, we use the explanation information to guided.  We also use the random motifs to guided adversarial samples generation. As the results show, LM$^2$otifs is more robust than the baselines. 

> **5. Additionally, the writing should be improved. Please double check the syntax, symbols, and formulas. For example, on line 122, the definition for accuracy, and for equation 1.**

To be continued.



> 6. The method makes sense for the problem. But it can be difficult for an application. It seems that this method is suitable for judging a collection of documents. For judging a single document, it is not easy to do. The author did not mention this problem.

Thanks for pointing out the application. Our method is easy to use for a single document. In LM$^2$otifs, the dimension of node feature is based on the tokens in the training set. We initialize the token nodes with one-hot features and the documents node with an all-zeros feature. For the graph contains different documents, we don't need to change the GNN model. For a single document, once the document node is connected with token nodes, LM$^2$otifs works. So our method is not limited in a collection of documents.

> **7. In Theorem 3.1, the authors mention assumption “the human-generated text always ends with the token a and the machine-generated text with token b”, this is not possible in the real-world setting.**

To be continued.





> 8. But the efficiency of the graph construction is not mentioned anymore. And what are the advantages? Only Interpretable results？Some methods like threshold-based can also clearly show the probability distribution of one sentence.

Thanks for the insightful question.The graph construction is depended on the length of the document $L$ and the length of the sliding window $W$, the time complexity is $O(LW^2)$. In this paper, we are exploring if the PGM-based methods can be used for authorship detections. We support this claim from theortical proof and empirical experiments. Compared with zero-shot methods, our method is a training-based method providing more accurate results and do not rely on commercial api. Comparing to the training-based methods, LM$^2$otifs requires limited training samples, which is beneficial to detect a new LLM. 
    
Besides, the goal of LM$^2$otifs framework is to extract the explanation motifs insight the HGTs and MGTs, such as the perference words and phrase of LLMs, instead of explaining the methodology. LM$^2$otifs, PGM baed method can easily to extract different fine granularity motifs by controlling the explanation subgraphs for both global and local patterns, while traditional method is hard to analyze. For example, the attention based explanation method is easy to obtain the local explanations but hard to capture the global pattern of one LLM.

The probability distribution of documents provided by threshold-based methods do not reveal the reason why the distribution should be like, lacking of support evidence.
 
> 9. Strengths And Weaknesses:

These concerns are answered in reply 6 and 8.