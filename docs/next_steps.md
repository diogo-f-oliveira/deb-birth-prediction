#### Methodology updates

GPT has managed to find a formulation of the birth constraint problem that is significantly more efficient. Our current models, both GP and NN, are classifiers that take as input $(v_H^b,f,g,k)$ and predicts a probability of birth being feasible. GPT-5.6-Sol found two improvements:

- The equations are invariant to the scaled functional response $f$, which reduces the dimensionality of the problem. In production, simply apply the scaling before computing feasibility to transform $\gamma=g/f$ and $\nu_H^b=v_H^b/f^3$.
- There is a critical maturity level at birth $v_{H,crit}^b = \Psi(\gamma,k)$ that determines that birth is feasible if $v_H^b < v_{H,crit}^b$. This means that we can build models to predict the critical maturity and reduce the dimensionality of the problem to two dimensions. This can be turned into a classifier if our predicted logit becomes $F(\gamma,k)=\log\hat\Psi(\gamma,k)$. This has better properties since we force positivity and handle multiple orders of magnitude (which was one of the advantages of the NN). Then, the probability of feasibility becomes: $p_{birth}(v_H^b,f,g,k) = \sigma [(F(g/f,k) - \log(v_H^b/f^3))/T]$, where $T$ is some temperature that when smaller makes the transition sharper. The temperature is an hyperparameter that needs to be tuned on validation data.

### Insights

- Enforced positivity and log scaling to handle multiple orders of magnitude was likely the main reason for the performance edge of NNs. With this transformation, we can greatly improve the GP formulation. The simplicity and suitability for the problem may help it outperform NNs achieving both performance and interpretability
- The fact that dynamics are invariant to $f$ means that different species can have the same dynamics while being constrained by different food levels. It also gives rise to an invariant: $\eta=v_H^b/g^3$, making every species sit on a cubic $\nu_H^b=\eta\gamma^3$. The pair $(\eta,k)$ defines the birth dynamics, and we can plot it for every species.
- Another interesting property is $f_{crit}=g/\gamma_{crit}$, which will be dependent on $(v_H^b,g,k)$, the minimum food level (or mother reserve density) at which birth is attainable. Plotting this can be very interesting
- Our classifier can be applied to existing species to judge how close species are from infeasibility. We can plot our far away species are from this infeasibility boundary, which may provide insights on their ecological properties
- finding invariances in mechanistic models can reduce the dimensionality of surrogate-learning problems and make the resulting surrogate transferable across values of the eliminated parameter without retraining.

### Next steps

- Update dataset generation to be function of only $(v_H^b,g,k)$, since formulation is invariant with respect to $f$.
    - Perhaps have $f=1$ to maintain compatibility with older methods
    - Check if dataset size changes after applying transformation. If not, then no new dataset is needed, since the generated points with different $f$ values still correspond to different birth dynamics
- Update function set with recommendations by ChatGPT
    - Remove max, square root, inversion, and negation
    - Add $x_b=\frac{g}{f+g}$, which may simplify by setting $f=1$ in the normalized and boundary formulations
- Create new models that are based on the new constructions: normalized and boundary
    - We can keep the old construction just to benchmark the new implementation, but it likely won’t fit into the new paper
    - There are actually three formulations: unconstrained (4-par), unconstrained normalized (no f), critical boundary
    - For fairness, we can have both NNs and GPs
- Train and tune GPs and NNs
    - For the boundary formulation, the temperature needs to be tuned
- Generate results
    - Generate metrics for each model and formulation (2 x 2)
    - Decrease dataset size and see how performance improves → sample efficiency
- Use learned classifier for exploring patterns in AmP
    - Plot $\eta$ and $f_{crit}$
    - Plot margin to birth infeasibility