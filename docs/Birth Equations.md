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

Prior to growth cessation, \(W(s)<W(\tau)\) for \(s<\tau\). This yields three qualitatively different regimes.

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

For \(k>1\), maturation ceases before the growth limit is reached. Let \(\lambda_R<1\) denote the first terminal length for which maturation is exactly stationary at birth. Evaluating Eq. (12) at

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