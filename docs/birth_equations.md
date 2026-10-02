## Birth feasibility as a critical-maturity boundary

### Scaled embryonic dynamics

During embryonic development, assimilation is zero and the organism develops exclusively from the reserve initially deposited in the egg. Let \(\tau\) denote scaled time, \(l\) scaled structural length, \(e\) scaled reserve density, and \(u_H\) scaled maturity. Define

\[
v_H=\frac{u_H}{1-\kappa},
\]

and let

\[
g>0
\]

be the energy investment ratio and

\[
k=\frac{k_J}{k_M}>0
\]

the ratio between maturity- and somatic-maintenance rate coefficients.

The embryonic dynamics can then be written as

\[
\frac{de}{d\tau}
=
-g\frac{e}{l},
\tag{1}
\]

\[
\frac{dl}{d\tau}
=
\frac{g}{3}\frac{e-l}{e+g},
\tag{2}
\]

and

\[
\frac{dv_H}{d\tau}
=
\frac{e\,l^2(g+l)}{e+g}
-kv_H.
\tag{3}
\]

At the start of development,

\[
l(0)=0,
\qquad
v_H(0)=0,
\qquad
e(0)=+\infty.
\tag{4}
\]

The divergence of \(e\) results from reserve density being defined relative to vanishing initial structural volume and does not imply that the initial amount of reserve is infinite.

Under the standard maternal-effect assumption, the scaled reserve density at birth equals the scaled functional response of the mother,

\[
e_b=f,
\tag{5}
\]

whereas the maturity threshold for birth is a model parameter,

\[
v_H(\tau_b)=v_H^b.
\tag{6}
\]

The structural length at birth, \(l_b\), and age at birth, \(\tau_b\), are outcomes of the embryonic dynamics rather than independent parameters.

For birth to be reached while embryonic development remains viable, growth and maturation must still be positive at the birth boundary. From Eqs. (2) and (3), these conditions are

\[
l_b<f,
\tag{7}
\]

and

\[
kv_H^b
<
l_b^2
\frac{(g+l_b)f}{g+f}.
\tag{8}
\]

The objective is to determine whether these conditions can be satisfied without first solving for \(l_b\).

**Assumptions and definition of feasibility.** These are made explicit by the T07 review, and the proofs are in the section *Proof of the critical-boundary characterization* below.

- **Domain.** The parameters are finite with \(g,k,f,v_H^b>0\) and \(0<\kappa<1\). Neither \(f\le1\) nor \(k\le1\) is assumed.
- **Eggs.** An egg is a solution of Eqs. (1)-(3) started from the singular state (4) with a finite initial reserve \(u_E^0>0\). The singular state is understood through \(u_E^0\): see Lemma 1.
- **Birth.** Birth occurs the *first* time maturity reaches \(v_H^b\), and the maternal-effect rule (5) requires \(e=f\) at that instant.
- **Feasibility.** \((g,k,v_H^b,f)\) is feasible when some egg reaches birth in this sense with the strict conditions (7) and (8). Equality in (7) or (8) counts as infeasible.

Lemma 4 shows that the first-hit requirement adds nothing once (8) holds. Lemma 2 shows that (7) at birth implies positive growth throughout development.

---

## Elimination of the maternal food level

The four quantities \((g,k,v_H^b,f)\) appear to determine birth feasibility. However, \(f\) can be removed exactly by exploiting a scaling symmetry of the embryo equations.

Introduce

\[
\epsilon=\frac{e}{f},
\qquad
\lambda=\frac{l}{f},
\qquad
\gamma=\frac{g}{f},
\qquad
\nu=\frac{v_H}{f^3}.
\tag{9}
\]

Substituting

\[
e=f\epsilon,
\qquad
l=f\lambda,
\qquad
g=f\gamma,
\qquad
v_H=f^3\nu
\]

into Eqs. (1)--(3) gives

\[
\frac{d\epsilon}{d\tau}
=
-\gamma\frac{\epsilon}{\lambda},
\tag{10}
\]

\[
\frac{d\lambda}{d\tau}
=
\frac{\gamma}{3}
\frac{\epsilon-\lambda}{\epsilon+\gamma},
\tag{11}
\]

and

\[
\frac{d\nu}{d\tau}
=
\frac{
\epsilon\lambda^2(\gamma+\lambda)
}{
\epsilon+\gamma
}
-k\nu.
\tag{12}
\]

The maternal food level has disappeared completely.

At birth,

\[
\epsilon_b=1,
\qquad
\lambda_b=\frac{l_b}{f},
\qquad
\nu_b=\frac{v_H^b}{f^3}.
\tag{13}
\]

Consequently, the embryonic boundary-value problem with arbitrary \(f\) is exactly equivalent to one with \(f=1\) and transformed parameters

\[
\boxed{
\gamma=\frac{g}{f},
\qquad
k=k,
\qquad
\nu_b=\frac{v_H^b}{f^3}.
}
\tag{14}
\]

Birth feasibility therefore cannot depend independently on \(g\), \(v_H^b\), and \(f\). It can depend only on

\[
\frac{g}{f},
\qquad
k,
\qquad
\frac{v_H^b}{f^3}.
\tag{15}
\]

This reduces the effective dimension of the feasibility problem by one.

---

## Structural trajectories terminating at a prescribed birth length

To characterize which maturity levels are attainable, consider a candidate normalized birth length

\[
0<\lambda_b\le 1.
\]

Because \(\epsilon\) decreases monotonically during embryonic development, it can be used as the independent variable. Combining Eqs. (10) and (11),

\[
\frac{d\lambda}{d\epsilon}
=
-
\frac{
\lambda(\epsilon-\lambda)
}{
3\epsilon(\epsilon+\gamma)
}.
\tag{16}
\]

Introducing

\[
q=\frac{1}{\lambda}
\]

transforms this nonlinear equation into the linear equation

\[
\frac{dq}{d\epsilon}
-
\frac{q}{3(\epsilon+\gamma)}
=
-
\frac{1}{
3\epsilon(\epsilon+\gamma)
}.
\tag{17}
\]

An integrating factor is

\[
(\epsilon+\gamma)^{-1/3}.
\]

Using the terminal condition

\[
\lambda(1)=\lambda_b,
\]

integration gives

\[
\boxed{
\frac{1}{\lambda(\epsilon)}
=
(\epsilon+\gamma)^{1/3}
\left[
\frac{1}{
\lambda_b(1+\gamma)^{1/3}
}
-
\frac{1}{3}
\int_1^\epsilon
\frac{ds}{
s(s+\gamma)^{4/3}
}
\right].
}
\tag{18}
\]

Thus, for fixed \((\gamma,\lambda_b)\), the complete structural trajectory is known without solving the original coupled system.

The integral in Eq. (18) can alternatively be expressed using incomplete beta or hypergeometric functions, but this representation is more useful for the present feasibility analysis.

---

## Existence of a finite egg solution

It is also possible to show directly that every candidate

\[
0<\lambda_b\le1
\]

corresponds to a finite initial amount of reserve.

Define

\[
A(\lambda_b,\gamma)
=
\frac{1}{
\lambda_b(1+\gamma)^{1/3}
}
-
\frac13
\int_1^\infty
\frac{ds}{
s(s+\gamma)^{4/3}
}.
\tag{19}
\]

Since \(s>1\),

\[
\frac13
\int_1^\infty
\frac{ds}{
s(s+\gamma)^{4/3}
}
<
\frac13
\int_1^\infty
\frac{ds}{
(s+\gamma)^{4/3}
}
=
\frac{1}{
(1+\gamma)^{1/3}
}.
\tag{20}
\]

For \(\lambda_b\le1\),

\[
\frac{1}{
\lambda_b(1+\gamma)^{1/3}
}
\ge
\frac{1}{
(1+\gamma)^{1/3}
},
\]

and therefore

\[
A(\lambda_b,\gamma)>0.
\tag{21}
\]

From Eq. (18),

\[
\lambda(\epsilon)
\sim
\frac{1}{
A\epsilon^{1/3}
}
\qquad
\text{as }
\epsilon\rightarrow\infty.
\tag{22}
\]

The scaled reserve amount satisfies

\[
e=\frac{gu_E}{l^3},
\]

which under the transformation in Eq. (9) becomes

\[
\frac{u_E}{f^3}
=
\frac{
\epsilon\lambda^3
}{
\gamma
}.
\tag{23}
\]

Consequently,

\[
\boxed{
\frac{u_E^0}{f^3}
=
\frac{1}{
\gamma A^3
}
<\infty.
}
\tag{24}
\]

The corresponding age at birth is obtained from Eq. (10),

\[
\tau_b
=
\frac{1}{\gamma}
\int_1^\infty
\frac{\lambda(\epsilon)}{\epsilon}
\,d\epsilon.
\tag{25}
\]

Because Eq. (22) gives an asymptotic integrand proportional to \(\epsilon^{-4/3}\), this integral also converges,

\[
\boxed{\tau_b<\infty.}
\tag{26}
\]

Therefore every normalized terminal length satisfying

\[
0<\lambda_b\le1
\]

defines a finite egg-development trajectory. The strict condition

\[
\lambda_b<1
\]

additionally ensures positive structural growth at birth.

---

## Maturity attained by each structural trajectory

For each candidate terminal length \(\lambda_b\), the structural trajectory in Eq. (18) determines the maturity accumulated before the reserve density reaches \(\epsilon=1\).

Combining Eqs. (10) and (12) gives

\[
\frac{d\nu}{d\epsilon}
-
\frac{
k\lambda
}{
\gamma\epsilon
}\nu
=
-
\frac{
\lambda^3(\gamma+\lambda)
}{
\gamma(\epsilon+\gamma)
}.
\tag{27}
\]

This is a first-order linear equation with

\[
\nu(\infty)=0.
\]

Its solution evaluated at birth defines the attainable normalized maturity

\[
\nu_b
=
\Phi(\lambda_b;\gamma,k),
\]

where

\[
\boxed{
\Phi(\lambda_b;\gamma,k)
=
\int_1^\infty
\frac{
\lambda(\epsilon)^3
[\gamma+\lambda(\epsilon)]
}{
\gamma(\epsilon+\gamma)
}
\exp
\left[
-\frac{k}{\gamma}
\int_1^\epsilon
\frac{\lambda(r)}{r}\,dr
\right]
d\epsilon.
}
\tag{28}
\]

Here \(\lambda(\epsilon)\) is the structural trajectory from Eq. (18).

Thus, for fixed \((\gamma,k)\), varying \(\lambda_b\) generates a continuous family of finite egg trajectories and corresponding attainable maturities,

\[
\lambda_b
\longmapsto
\Phi(\lambda_b;\gamma,k).
\tag{29}
\]

---

## Which process limits embryonic development?

The critical boundary can be characterized further by comparing growth and maturation.

Define normalized structural volume

\[
W=\lambda^3.
\]

From Eq. (11),

\[
\frac{dW}{d\tau}
=
\gamma\lambda^2
\frac{
\epsilon-\lambda
}{
\epsilon+\gamma
}.
\tag{30}
\]

Subtracting this expression from the maturity-production term in Eq. (12) gives the exact identity

\[
\boxed{
\frac{d\nu}{d\tau}
=
\frac{dW}{d\tau}
+
W
-k\nu.
}
\tag{31}
\]

Let

\[
z=\nu-W.
\]

Then

\[
\frac{dz}{d\tau}
+kz
=
(1-k)W,
\]

and since \(z(0)=0\),

\[
\boxed{
\nu(\tau)
=
W(\tau)
+
(1-k)
\int_0^\tau
e^{-k(\tau-s)}
W(s)\,ds.
}
\tag{32}
\]

Prior to growth cessation, \(W(s)<W(\tau)\) for \(s<\tau\). For every egg with \(\lambda_b\le1\), this holds on the whole interval \((0,\tau_b]\) by Lemma 2 below. This yields three qualitatively different regimes.

For

\[
0<k<1,
\]

\[
W<\nu<\frac{W}{k}.
\tag{33}
\]

For

\[
k=1,
\]

\[
\nu=W.
\tag{34}
\]

For

\[
k>1,
\]

\[
\frac{W}{k}<\nu<W.
\tag{35}
\]

At the instant at which growth ceases,

\[
\frac{dW}{d\tau}=0,
\]

so Eq. (31) becomes

\[
\frac{d\nu}{d\tau}
=
W-k\nu.
\tag{36}
\]

Therefore,

\[
\begin{cases}
d\nu/d\tau>0, & 0<k<1,\\[1mm]
d\nu/d\tau=0, & k=1,\\[1mm]
d\nu/d\tau<0, & k>1.
\end{cases}
\tag{37}
\]

Hence the two viability constraints have a clear interpretation:

- for \(k<1\), **growth is limiting first**;
- for \(k=1\), growth and maturation become limiting simultaneously;
- for \(k>1\), **maturation is limiting first**.

---

## Critical normalized maturity

The largest normalized maturity threshold that can be reached by a viable embryo can now be defined as a function of only \((\gamma,k)\),

\[
\boxed{
\Psi(\gamma,k)
=
\nu_{b,\mathrm{crit}}.
}
\tag{38}
\]

For \(0<k<1\), growth remains the limiting process. The viability boundary is reached when

\[
\lambda_b\rightarrow1,
\]

and therefore

\[
\boxed{
\Psi(\gamma,k)
=
\Phi(1;\gamma,k),
\qquad
0<k<1.
}
\tag{39}
\]

For \(k=1\), Eq. (34) gives

\[
\nu_b=\lambda_b^3,
\]

and the critical value occurs at \(\lambda_b=1\),

\[
\boxed{
\Psi(\gamma,1)=1.
}
\tag{40}
\]

Eq. (39) uses the fact that \(\Phi(\lambda_b;\gamma,k)<\Phi(1;\gamma,k)\) for every \(\lambda_b<1\) and that every \(\lambda_b<1\) is viable. Both facts are proved in Theorem 1 below.

For \(k>1\), maturation ceases before the growth limit is reached. Let \(\lambda_R<1\) denote the first terminal length for which maturation is exactly stationary at birth. Lemma 6 below proves that it exists, and Lemma C proves that it is the only maturation-stationary terminal length in \((0,1)\). Evaluating Eq. (12) at

\[
\epsilon_b=1
\]

gives

\[
k\Phi(\lambda_R;\gamma,k)
=
\lambda_R^2
\frac{
\gamma+\lambda_R
}{
1+\gamma
}.
\tag{41}
\]

The critical maturity is therefore

\[
\boxed{
\Psi(\gamma,k)
=
\Phi(\lambda_R;\gamma,k)
=
\frac{
\lambda_R^2(\gamma+\lambda_R)
}{
k(1+\gamma)
},
\qquad
k>1.
}
\tag{42}
\]

Thus the full embryonic feasibility problem reduces to a two-dimensional critical surface,

\[
\boxed{
\nu_{b,\mathrm{crit}}
=
\Psi(\gamma,k).
}
\tag{43}
\]

Birth is reachable precisely when

\[
\boxed{
\nu_b<\Psi(\gamma,k).
}
\tag{44}
\]

**Proof status (T07).**

- **\(k\le1\):** Eq. (44) is proved in both directions (Theorem 1).
- **\(k>1\):** Eq. (44) is proved in both directions (Theorem 2). Lemma C shows that \(\lambda_R\) is the unique zero of the maturation rate at birth, so no egg larger than \(\lambda_R\) is viable.

Returning to the original variables,

\[
\nu_b=\frac{v_H^b}{f^3},
\qquad
\gamma=\frac{g}{f},
\]

so

\[
\boxed{
v_H^b
<
f^3
\Psi\left(
\frac{g}{f},k
\right).
}
\tag{45}
\]

Equation (45) is the desired parameter-only representation of birth feasibility. The function \(\Psi\) is universal for the standard DEB embryo equations: changing \(f\) merely rescales the arguments and output.

The difficulty is therefore no longer whether such a boundary exists, but whether \(\Psi\) admits a sufficiently simple analytical approximation.

---

## Proof of the critical-boundary characterization

This section was added by the T07 review on 2026-10-02. It supplies the arguments that the derivation above leaves implicit. All statements use the feasibility definition given after Eq. (8). The parameters satisfy \(\gamma>0\) and \(k>0\).

The review rechecked the algebra of Eqs. (10)-(12), (16)-(18), (20), (22)-(28) and (30)-(35) and found it correct.

The proofs below are analytical. Numerical computation is not used as evidence.

### Notation

\[
I(\epsilon)=\frac13\int_1^\epsilon\frac{ds}{s(s+\gamma)^{4/3}},\qquad
I_\infty=I(\infty)<(1+\gamma)^{-1/3},
\tag{P1}
\]

where the bound is Eq. (20). Along a trajectory, the remaining time to birth, the maturation source, the maturation rate and the stationarity function are

\[
\sigma(\epsilon)=\frac1\gamma\int_1^\epsilon\frac{\lambda(r)}{r}\,dr,\qquad
S=\frac{\epsilon\lambda^2(\gamma+\lambda)}{\epsilon+\gamma},\qquad
y=\frac{d\nu}{d\tau}=S-k\nu,
\tag{P2}
\]

\[
R(\lambda_b)=\frac{\lambda_b^2(\gamma+\lambda_b)}{1+\gamma},\qquad
D(\lambda_b)=R(\lambda_b)-k\Phi(\lambda_b;\gamma,k).
\tag{P3}
\]

\(S\) is the maturation source in Eq. (12), and \(R(\lambda_b)\) is its value at birth. \(D(\lambda_b)\) is the maturation rate \(y\) at birth, so condition (8) reads \(D(\lambda_b)>0\). The set of *viable* terminal lengths is

\[
V(\gamma,k)=\{\lambda_b\in(0,1):D(\lambda_b)>0\},
\]

and \(\mathcal F(\gamma,k)\) denotes the set of feasible \(\nu_b\).

### Lemma 1 (each egg corresponds to exactly one terminal length)

**(a) Positive solutions of (16).** Every positive solution of Eq. (16) on \([1,\infty)\) has the form

\[
1/\lambda=(\epsilon+\gamma)^{1/3}[C-I(\epsilon)]
\]

with a constant \(C\ge I_\infty\), so it is Eq. (18) with \(\lambda_b=1/[(1+\gamma)^{1/3}C]\). Part (b) shows that a finite egg requires \(C>I_\infty\).

**(b) Identification of the egg.** By Eq. (23), \(\epsilon\lambda^3=\gamma u_E/f^3\). Its limit as \(\epsilon\to\infty\) is \(1/A^3=\gamma u_E^0/f^3\), where

\[
A=C-I_\infty=\frac{1}{\lambda_b(1+\gamma)^{1/3}}-I_\infty .
\]

The reparametrization is valid because \(d\epsilon/d\tau=-\gamma\epsilon/\lambda<0\), with \(\tau=0\) corresponding to \(\epsilon=\infty\). Each finite initial reserve \(u_E^0\) therefore determines exactly one trajectory.

The map \(\lambda_b\mapsto A\) is a strictly decreasing bijection from \((0,\lambda_{\max})\) onto \((0,\infty)\), where \(\lambda_{\max}=[(1+\gamma)^{1/3}I_\infty]^{-1}>1\) by (P1). Eggs with \(\lambda_b\ge1\) violate (7). The candidate eggs are therefore exactly \(\lambda_b\in(0,1)\), and they are distinct eggs.

**(c) Ordering of trajectories.** Pointwise in \(\epsilon\),

\[
\mu:=\frac{\partial\lambda}{\partial\lambda_b}=\frac{\lambda^2}{\lambda_b^2}\Big(\frac{\epsilon+\gamma}{1+\gamma}\Big)^{1/3}>0 .
\tag{P4}
\]

**(d) Maturity.** The condition \(\nu(0)=0\) becomes \(\nu\to0\) as \(\epsilon\to\infty\). The homogeneous solution of Eq. (27) is proportional to \(e^{k\sigma(\epsilon)}\). It tends to \(e^{k\tau_b}\neq0\), so it is excluded. The unique solution is

\[
\nu(\epsilon)=\int_\epsilon^\infty
\frac{\lambda(s)^3[\gamma+\lambda(s)]}{\gamma(s+\gamma)}
e^{-k[\sigma(s)-\sigma(\epsilon)]}\,ds ,
\tag{P5}
\]

and Eq. (28) is \(\nu(1)\).

### Lemma 2 (growth is positive throughout development)

If \(\lambda_b\le1\), then \(\lambda(\epsilon)<\epsilon\) for every \(\epsilon>1\). Hence \(d\lambda/d\tau>0\) and \(dW/d\tau>0\) on \((0,\tau_b)\).

*Proof.* Let \(\delta(\epsilon)=\epsilon-\lambda(\epsilon)\). By Eq. (16), \(d\lambda/d\epsilon=0\) wherever \(\delta=0\), so \(d\delta/d\epsilon=1\) at every zero of \(\delta\). This means \(\delta\) can only cross zero upward as \(\epsilon\) increases. Since \(\delta(1)=1-\lambda_b\ge0\), there is no zero of \(\delta\) on \((1,\infty)\).

In more detail: if \(\lambda_b<1\), the first zero above 1 would be reached from positive values, which requires \(d\delta/d\epsilon\le0\) there. If \(\lambda_b=1\), \(\delta\) is positive immediately above 1, and the same argument applies. \(\square\)

This justifies the assumption that \(W(s)<W(\tau)\) for \(s<\tau\), which is used for Eqs. (33)-(35).

### Lemma 3 (bounds and regularity of \(\Phi\))

For \(\lambda_b\in(0,1]\), Eqs. (32)-(35) at \(\tau=\tau_b\), where \(W=\lambda_b^3\), give

\[
\lambda_b^3<\Phi<\lambda_b^3/k\ \ (k<1),\qquad
\Phi=\lambda_b^3\ \ (k=1),\qquad
\lambda_b^3/k<\Phi<\lambda_b^3\ \ (k>1).
\tag{P6}
\]

Hence \(\Phi\to0\) as \(\lambda_b\to0\).

For \(\lambda_b\le1\), the bracket in Eq. (18) is at least \(A(1,\gamma)>0\). This gives the uniform bound \(\lambda\le A(1,\gamma)^{-1}(\epsilon+\gamma)^{-1/3}\). Together with \(\lambda\le\lambda_b\le1\) and \(e^{-k\sigma}\le1\), it dominates the integrand of Eq. (28) by a multiple of \((\epsilon+\gamma)^{-2}\). It also dominates \(\mu/\epsilon\) by a multiple of \(\epsilon^{-4/3}\), locally uniformly in \(\lambda_b\).

By dominated convergence, \(\Phi\) is continuous on \((0,1]\) and continuously differentiable there, with one-sided differentiability at 1. Differentiation under the integral sign is justified by the same bounds.

### Lemma 4 (maturity has at most one maximum)

Differentiating \(S\) along a trajectory gives

\[
\frac{dS}{d\tau}=\frac{\gamma\epsilon\lambda}{(\epsilon+\gamma)^2}\,Q,\qquad
3Q=(2\gamma+3\lambda)(\epsilon-\lambda)-3\gamma(\gamma+\lambda).
\tag{P7}
\]

At any zero of \(Q\), we have \(\epsilon-\lambda=3\gamma(\gamma+\lambda)/(2\gamma+3\lambda)\). Substituting this,

\[
3\frac{dQ}{d\tau}
=(2\gamma+3\lambda)\frac{d\epsilon}{d\tau}
-\frac{\gamma^2+12\gamma\lambda+9\lambda^2}{2\gamma+3\lambda}\,\frac{d\lambda}{d\tau}<0,
\]

because \(d\epsilon/d\tau<0\) and, by Lemma 2, \(d\lambda/d\tau>0\). Also \(Q\to+\infty\) as \(\tau\to0\). Therefore \(Q\) changes sign at most once, from positive to negative, and \(S\) increases and then possibly decreases.

Since \(S\to0\) and \(\nu\to0\) as \(\tau\to0\), the maturation rate is

\[
y(\tau)=\int_0^\tau e^{-k(\tau-s)}\frac{dS}{ds}\,ds .
\]

So \(y>0\) while \(S\) is increasing. Once \(y(\tau_1)\le0\) at a time when \(S\) is decreasing,

\[
y(\tau)=e^{-k(\tau-\tau_1)}y(\tau_1)+\int_{\tau_1}^\tau e^{-k(\tau-s)}\,dS(s)<0
\]

for all \(\tau>\tau_1\). Hence \(y\) changes sign at most once, from positive to negative, and \(\nu\) has at most one maximum.

**Consequences.**

- If \(y(\tau_b)\ge0\), then \(y>0\) on \((0,\tau_b)\). In particular, condition (8) at birth implies that maturity increases strictly throughout development. The first time maturity reaches \(\nu_b\) is therefore \(\tau_b\), so the first-hit requirement is automatic.
- If \(D(\lambda_b)<0\), then \(\nu(1)\) was already exceeded before \(\epsilon=1\), so it is not a first hit.

Combined with Lemmas 1 and 2, this gives

\[
\mathcal F(\gamma,k)=\Phi\big(V(\gamma,k);\gamma,k\big).
\tag{P8}
\]

### Lemma 5 (\(\Phi\) increases with terminal length on viable trajectories)

Differentiate Eq. (28), using (P4). In the term coming from \(\partial\sigma/\partial\lambda_b=\gamma^{-1}\int_1^\epsilon\mu(r)\,dr/r\), exchange the order of integration (Fubini, for nonnegative integrands) and use (P5). This gives

\[
\frac{\partial\Phi}{\partial\lambda_b}
=\frac1\gamma\int_1^\infty e^{-k\sigma(\epsilon)}\,\frac{\mu(\epsilon)}{\epsilon}
\left[S\,\frac{3\gamma+4\lambda}{\gamma+\lambda}-k\nu\right]d\epsilon .
\tag{P9}
\]

Because \((3\gamma+4\lambda)/(\gamma+\lambda)\ge3\), the bracket is at least \(3S-k\nu=2S+y\). If \(y\ge0\) along the entire trajectory, every factor is positive and \(\partial\Phi/\partial\lambda_b>0\). By Lemma 4, this holds for every \(\lambda_b\) with \(D(\lambda_b)\ge0\).

It follows that \(\Phi\) is strictly increasing on every interval of terminal lengths on which \(D\ge0\).

### Theorem 1 (the case \(0<k\le1\))

\(\mathcal F(\gamma,k)=(0,\Phi(1;\gamma,k))\), so \(\Psi(\gamma,k)=\Phi(1;\gamma,k)\), which is Eq. (39). In particular \(\Psi(\gamma,1)=1\), which is Eq. (40). Each feasible \(\nu_b\) is produced by exactly one egg, and \(\nu_b=\Psi\) is infeasible.

*Proof.* By Eq. (31), \(y=dW/d\tau+W-k\nu\).

- **Sign of \(y\).** By Lemma 2, \(dW/d\tau\ge0\) on \((0,\tau_b]\) for \(\lambda_b\le1\). It is strictly positive at \(\tau_b\) when \(\lambda_b<1\). By Eqs. (33)-(34), \(W-k\nu>0\) for \(k<1\) and \(W-k\nu=0\) for \(k=1\). Hence \(y>0\) on \((0,\tau_b]\) for every \(\lambda_b\in(0,1)\), which gives \(V=(0,1)\). For \(\lambda_b=1\), \(y\ge0\) throughout.
- **Image of \(\Phi\).** By Lemma 5, \(\Phi\) is strictly increasing on \((0,1]\). By Lemma 3, it is continuous and tends to 0. So \(\Phi\big((0,1)\big)=(0,\Phi(1))\), and (P8) gives the claim.
- **Equality.** The only egg with \(\Phi=\Phi(1)\) is \(\lambda_b=1\). It has zero growth at birth and violates the strict condition (7); for \(k=1\) it also violates (8).
- **The case \(k=1\).** Here \(\Phi=\lambda_b^3\) by (P6). \(\square\)

### Lemma 6 (existence of \(\lambda_R\) for \(k>1\))

Let

\[
\lambda_{\mathrm{low}}=\frac{\gamma}{k(1+\gamma)-1}\in(0,1).
\]

By (P6), \(D(\lambda_b)>\lambda_b^2\left[(\gamma+\lambda_b)/(1+\gamma)-k\lambda_b\right]\ge0\) for \(\lambda_b\le\lambda_{\mathrm{low}}\). At the other end, \(D(1)=1-k\Phi(1)<0\), again by (P6).

\(D\) is continuous (Lemma 3), so its zero set in \([\lambda_{\mathrm{low}},1)\) is nonempty and closed. Define \(\lambda_R\) as its minimum. Then \(\lambda_{\mathrm{low}}<\lambda_R<1\), \((0,\lambda_R)\subseteq V\), and \(D(\lambda_R)=0\), which is Eq. (41). \(\square\)

### Theorem 2 (the case \(k>1\))

Let \(\Psi=\Phi(\lambda_R)=R(\lambda_R)/k\), as in Eq. (42).

**(a) Sufficiency.** Every \(\nu_b\in(0,\Psi)\) is feasible. It is produced by exactly one egg in \((0,\lambda_R)\).

*Proof.* \(D\ge0\) on \((0,\lambda_R]\), so by Lemma 5 \(\Phi\) is strictly increasing there. By Lemma 3, \(\Phi\big((0,\lambda_R)\big)=(0,\Psi)\), and these eggs lie in \(V\). \(\square\)

**(b) Equality.** Within \((0,\lambda_R]\), only the egg \(\lambda_R\) reaches \(\Psi\). On that egg, maturity touches \(\nu_b\) tangentially, with \(y=0\) at birth, which violates the strict condition (8).

**(c) Necessity.** By Lemma C below, \(V=(0,\lambda_R)\), so by (P8) \(\mathcal F=(0,\Psi)\). Eq. (44) therefore holds in both directions, each feasible \(\nu_b\) is produced by exactly one egg, and \(\nu_b=\Psi\) is infeasible.

### Lemma C (\(\lambda_R\) is the only zero of \(D\))

**Why it is needed.** By (P8), \(\mathcal F=\Phi(V)\). Suppose \(V\) had a component above \(\lambda_R\) on which \(\Phi\ge\Psi\). Then feasible maturities would exist above \(\Psi\), and the "first" stationary length in Eq. (41) would not give the critical value. Lemma C rules this out. It shows that the root of \(D\) is unique, so the selection "first \(\lambda_R\)" is automatic.

**Claim.** For \(k\ge1\) and every \(\lambda_b\in(0,1)\) with \(D(\lambda_b)\ge0\),

\[
\frac{d}{d\lambda_b}\left[\frac{D(\lambda_b)}{\lambda_b^2(2\gamma+3\lambda_b)}\right]<0 .
\tag{P10}
\]

Consequently, for \(k>1\), \(\lambda_R\) is the only zero of \(D\) in \((0,1)\), and \(V=(0,\lambda_R)\).

*Proof of the consequence.* By (P10), \(D'(\lambda_R)<0\), so \(D<0\) just above \(\lambda_R\). Suppose \(D\ge0\) somewhere in \((\lambda_R,1)\). Let \(\lambda_2\) be the infimum of such points. Then \(\lambda_2>\lambda_R\), \(D(\lambda_2)=0\), and \(D<0\) on \((\lambda_R,\lambda_2)\), so \(D'(\lambda_2)\ge0\). This contradicts (P10) at \(\lambda_2\). \(\square\)

*Proof of (P10).*

**Coordinates.** Label trajectories by the constant \(C\) of Lemma 1(a), so that \(\lambda=\lambda(\epsilon,C)\) and

\[
\lambda_C:=\partial_C\lambda=-\lambda^2(\epsilon+\gamma)^{1/3}<0 .
\]

Write the maturity (P5) as \(N(\epsilon,C)\) and the maturation rate as \(Y(\epsilon,C)=S(\epsilon,\lambda)-kN\). Define

\[
Y_\lambda:=\partial_CY/\lambda_C ,
\]

the change of the maturation rate across trajectories at fixed \(\epsilon\), per unit change of \(\lambda\). At \(\epsilon=1\) we have \(\lambda=\lambda_b\) and \(d\lambda_b/dC=\lambda_C(1,C)\). Hence \(D(\lambda_b)=Y(1,C)\) and \(D'(\lambda_b)=Y_\lambda(1,C)\).

**Evolution along a trajectory.** Let a prime denote \(\partial_\epsilon\) at fixed \(C\), and let

\[
P=-\frac{dS}{d\epsilon}\Big|_{\text{trajectory}}=\frac{\lambda^2Q}{(\epsilon+\gamma)^2},
\]

which follows from (P7) and \(d\tau/d\epsilon=-\lambda/(\gamma\epsilon)\). Then \(\nu'=-(\lambda/\gamma\epsilon)\,y\) gives

\[
Y'=-P+\frac{k\lambda}{\gamma\epsilon}\,Y .
\tag{P11}
\]

Differentiate (P11) in \(C\), and use \(\lambda_C'/\lambda_C=(2\lambda-\epsilon)/(3\epsilon(\epsilon+\gamma))\), which follows from Eq. (16):

\[
Y_\lambda'=b\,Y_\lambda-P_\lambda+\frac{k}{\gamma\epsilon}\,Y,\qquad
b=\frac{k\lambda}{\gamma\epsilon}+\frac{\epsilon-2\lambda}{3\epsilon(\epsilon+\gamma)} .
\tag{P12}
\]

Here \(P_\lambda=\partial P/\partial\lambda\) at fixed \(\epsilon\).

**Weighted combination.** Let

\[
m(\lambda)=\partial_\lambda\log[\lambda^2(2\gamma+3\lambda)]=\frac2\lambda+\frac{3}{2\gamma+3\lambda},
\qquad W=Y_\lambda-m(\lambda)\,Y .
\]

Using \(\lambda'=-\lambda(\epsilon-\lambda)/(3\epsilon(\epsilon+\gamma))\), direct algebra gives

\[
W'=b\,W+\big(mP-P_\lambda\big)+E\,Y,
\tag{P13}
\]

with

\[
mP-P_\lambda=\frac{\lambda^2(\gamma^2+12\gamma\lambda+9\lambda^2)}{3(2\gamma+3\lambda)(\epsilon+\gamma)^2}>0,
\]

\[
E=\frac{4\gamma^2+6\gamma\epsilon+3\epsilon(2\gamma+3\lambda)^2/\gamma+3(k-1)(\epsilon+\gamma)(2\gamma+3\lambda)^2/\gamma}{3\epsilon(\epsilon+\gamma)(2\gamma+3\lambda)^2}>0\quad(k\ge1).
\]

Both forcing coefficients are positive regardless of the sign of \(Q\). This is why the weight \(m\) was chosen: it cancels every term of indefinite sign.

**Integration.** Let \(B(\epsilon)=\int_1^\epsilon b\). Integrating \((e^{-B}W)'=e^{-B}[(mP-P_\lambda)+EY]\) over \([1,\infty)\), and using \(e^{-B}W\to0\) (shown below), gives

\[
W(1)=-\int_1^\infty e^{-B(\epsilon)}\Big[(mP-P_\lambda)+E\,Y\Big]\,d\epsilon .
\]

If \(D(\lambda_b)\ge0\), Lemma 4 gives \(Y>0\) for \(\epsilon>1\), so the integrand is positive and \(W(1)<0\). Finally,

\[
W(1)=D'(\lambda_b)-m(\lambda_b)D(\lambda_b)=\lambda_b^2(2\gamma+3\lambda_b)\,\frac{d}{d\lambda_b}\left[\frac{D}{\lambda_b^2(2\gamma+3\lambda_b)}\right],
\]

which proves (P10).

**Behaviour as \(\epsilon\to\infty\).** Along a trajectory, \(\lambda\sim(A\epsilon^{1/3})^{-1}\). The terms \(k\lambda/(\gamma\epsilon)\) and \(\lambda/(\epsilon(\epsilon+\gamma))\) are integrable, so \(B=\tfrac13\log(\epsilon+\gamma)+O(1)\) and \(e^{-B}=O(\epsilon^{-1/3})\).

For \(W\):

- \(S=O(\epsilon^{-2/3})\) and \(N=O(\epsilon^{-1})\), so \(Y=O(\epsilon^{-2/3})\) and \(mY=O(\epsilon^{-1/3})\).
- \(\lambda_C\) is of exact order \(\epsilon^{-1/3}\), and \(S_\lambda\lambda_C=O(\epsilon^{-2/3})\).
- Differentiating (P5) in \(C\) under the integral gives \(N_C=O(\epsilon^{-1})\).
- Hence \(Y_\lambda=(S_\lambda\lambda_C-kN_C)/\lambda_C=O(\epsilon^{-1/3})\).

So \(W=O(\epsilon^{-1/3})\) and \(e^{-B}W\to0\).

The domination bounds of Lemma 3, locally uniform in \(C\), justify differentiating (P5) in \(C\) and exchanging \(\partial_C\) with \(\partial_\epsilon\). \(\square\)

**Checks.**

- The identities for \(P\), (P12), \(mP-P_\lambda\) and \(E\) were confirmed by symbolic algebra.
- At \(k=1\), the claim can be read off directly: \(D/[\lambda_b^2(2\gamma+3\lambda_b)]=\gamma(1-\lambda_b)/[(1+\gamma)(2\gamma+3\lambda_b)]\), which is decreasing.
- Before this proof was written, a throwaway numerical quadrature found no counterexample. It guided the search and is not part of the argument.

### Corollary 3 (necessary bounds and the sign of \(\log\Psi\))

**(a) Necessary condition.** Every feasible point satisfies \(k\nu_b<1\), that is, \(kv_H^b<f^3\). This follows from (7), (8) and the fact that \(R\) is increasing: \(k\nu_b<R(\lambda_b)<R(1)=1\).

The condition is not sufficient. For example, for \(k<1\) every \(\nu_b\in[\Phi(1),1/k)\) is infeasible, and this interval is nonempty by (P6).

The accepted CONTROLO'26 paper calls \(kv_H^b<f^3\) sufficient. This analysis shows it is necessary instead. The PDF is left unchanged.

**(b) Bounds on \(\Psi\).** From (P6) and Lemma 6:

\[
1<\Psi<\frac1k\ \ (k<1),\qquad
\Psi=1\ \ (k=1),\qquad
\lambda_{\mathrm{low}}^3<\Psi<\frac1k<1\ \ (k>1),
\tag{P14}
\]

For \(k>1\), the lower bound uses \(\Psi=R(\lambda_R)/k\), the fact that \(R\) is increasing, \(\lambda_R>\lambda_{\mathrm{low}}\), and \(R(\lambda_{\mathrm{low}})/k=\lambda_{\mathrm{low}}^3\). Hence

\[
\log\Psi \text{ has the sign of } 1-k,\qquad
\log\Psi\le-\log k,\ \text{with equality only at } k=1.
\]

These bounds are analytical constraints that a learned \(F\approx\log\Psi\) may be checked against.

### Ties and numerical tolerance

The boundary set \(\nu_b=\Psi\) is infeasible for all \(k\): by Theorem 1 for \(k\le1\), and by Theorem 2(b)-(c) for \(k>1\). This matches the strict decision rule \(F-\log\nu_b>0\) used by the implementation.

The practical labels (get_lb2 with a timeout and failures treated as infeasible) need not reproduce this surface exactly near the boundary.

### Status summary

| Step | Status |
| --- | --- |
| Scaling and integral expressions, Eqs. (9)-(35) | Checked |
| Egg ↔ \(\lambda_b\) bijection; maturity initial condition (Lemma 1) | Proved |
| Growth positive throughout development (Lemma 2) | Proved |
| First hit is implied by (8); single maturity maximum (Lemma 4) | Proved |
| Continuity of \(\Phi\), \(\Phi\to0\); monotonicity on viable eggs (Lemmas 3, 5) | Proved |
| \(k\le1\): \(\mathcal F=(0,\Psi)\), strict inequality, \(\Psi(\gamma,1)=1\) (Theorem 1) | Proved |
| \(k>1\): existence of \(\lambda_R\); \(\nu_b<\Psi\Rightarrow\) feasible (Lemma 6, Theorem 2a-b) | Proved |
| \(k>1\): unique zero \(\lambda_R\) of \(D\); feasible \(\Rightarrow\nu_b<\Psi\); egg uniqueness (Lemma C, Theorem 2c) | Proved |
| \(kv_H^b<f^3\) necessary; bounds and sign of \(\log\Psi\) (Corollary 3) | Proved |

---

# Theory-constrained genetic-programming classification

The preceding derivation substantially constrains the machine-learning problem.

An unconstrained classifier would attempt to learn

\[
C(v_H^b,g,k,f)
\rightarrow
\{0,1\}.
\]

However, theory shows that the true decision surface necessarily has the form

\[
\frac{v_H^b}{f^3}
=
\Psi\left(
\frac{g}{f},k
\right).
\]

The GP algorithm can therefore be restricted to learning only the unknown two-variable function \(\Psi\).

For numerical conditioning and to enforce positivity of the predicted critical maturity, it is convenient for GP to evolve

\[
\boxed{
F(\gamma,k)
\approx
\log\Psi(\gamma,k).
}
\tag{46}
\]

For each observation, define

\[
\gamma_i
=
\frac{g_i}{f_i},
\qquad
\nu_i
=
\frac{v_{H,i}^b}{f_i^3}.
\tag{47}
\]

The signed logarithmic distance from the predicted feasibility boundary is

\[
m_i
=
F(\gamma_i,k_i)
-
\log\nu_i
=
\log
\frac{
\widehat{\Psi}(\gamma_i,k_i)
}{
\nu_i
}.
\tag{48}
\]

Positive \(m_i\) corresponds to the predicted feasible region and negative \(m_i\) to the predicted infeasible region.

A probabilistic classifier is obtained by applying a sigmoid,

\[
\boxed{
\hat p_i
=
\sigma(\alpha m_i)
=
\frac{1}{
1+\exp(-\alpha m_i)
},
}
\tag{49}
\]

where

\[
\alpha>0
\]

controls the sharpness of the transition across the boundary.

At

\[
\hat p_i=0.5,
\]

the margin is zero and therefore

\[
\nu_i
=
\widehat{\Psi}(\gamma_i,k_i).
\]

Thus the GP expression retains the direct interpretation of a predicted critical maturity even though the model is trained as a classifier.

Training can use weighted binary cross-entropy,

\[
\mathcal L_{\mathrm{BCE}}
=
-
\frac1N
\sum_{i=1}^N
w_{y_i}
\left[
y_i\log\hat p_i
+
(1-y_i)\log(1-\hat p_i)
\right],
\tag{50}
\]

optionally combined with a parsimony penalty on GP tree size.

The resulting architecture is therefore

\[
(\gamma,k)
\xrightarrow{\mathrm{GP}}
F(\gamma,k)
\xrightarrow{\text{boundary comparison}}
F-\log\nu
\xrightarrow{\text{sigmoid}}
P(\text{birth}).
\tag{51}
\]

In the original parameterization,

\[
\boxed{
\hat p(\text{birth})
=
\sigma
\left\{
\alpha
\left[
F\left(\frac{g}{f},k\right)
-
\log\left(
\frac{v_H^b}{f^3}
\right)
\right]
\right\}.
}
\tag{52}
\]

The corresponding symbolic feasibility equation is

\[
\boxed{
v_H^b
<
f^3
\exp
\left[
F\left(
\frac{g}{f},k
\right)
\right].
}
\tag{53}
\]

This construction has several consequences. First, GP only evolves a function of two variables rather than an arbitrary expression involving four parameters. Second, \(v_H^b\) enters the classifier in the analytically prescribed manner and cannot be combined arbitrarily with the other inputs. Third, the exact scaling with maternal food level is imposed rather than learned from data. Fourth, the GP output itself remains interpretable as an approximation to the critical maturity surface.

The classifier probability can subsequently be used as a screening criterion during parameter estimation. Parameter sets classified with high confidence as feasible can proceed directly, those classified with high confidence as infeasible can be rejected, and parameter sets close to the predicted boundary can be passed to the full numerical birth solver. In this way the symbolic model acts as a fast surrogate while the numerical solution remains available for ambiguous cases.