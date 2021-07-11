---
author: "Malte Vo\\ss"
date: "\\today"
title: "Nonlinear Support Vector Machines"
subtitle: "Seminar Statistik -- Sommersemester 2021"
theme: "metropolis"
---

### Überblick

- Wiederholung: Lineare SVM
- *feature space*
- Kernel Trick
- SVM und Gradient descent

## Linear SVM

### Linear Support Vector Machines (SVM)

![Linear separierbare Daten mit Hyperebene als Diskriminator](../assets/linearly_separable_data.png){ width=50% }

- nutze separierende Funktion $f(x)=\beta_0+x^T\beta$

### Linear SVM -- Margin

- Hyperebene mit maximalen Margin wird gewählt

![Margin veranschaulicht](../assets/margin.png){ width=50% }

- Minimierungsproblem: $\min ||\beta||^2$ mit $y_i (\beta_0 + x_i^T \beta) \geq +1$

### Linear SVM -- Minimierungsproblem

- Minimierungsproblem: - $\min ||\beta||^2$ mit $y_i (\beta_0 + x_i^T \beta) \geq +1$
- Lösungsverfahren: Lagrange Multipliers
- Wolfe-Dual: $\max_\alpha 1_n^T \alpha - \dfrac{1}{2} \alpha^T H \alpha$ mit Nebenbedingungen $\alpha_i \geq 0, \alpha^T y = 0$

### Linear SVM -- linearly non-separable data

![Linear nicht separierbare Daten](../assets/linearly_non_separable_data.png){ width=45% }

- Verletzung des Margin wird erlaubt
- Gleichung wird um slack-Variablen erweitert
- $\min ||\beta||^2 + C\sum_i \xi_i$ mit $y_i (\beta_0 + x_i^T \beta) \geq +1 - \xi$ und $\xi \geq 0$

## Nonlinear SVM

### Linearly non-separable data: Two Moons

<!-- TODO: show two moons / other -->
![Data set: two moons](../assets/two_moons.png)

### Linearly non-separable data -- Goal

<!-- TODO: show two moons / other -->
![Data set: two moons with a separator](../assets/two_moons_nonlinear_separator.png)

### Nonlinear SVM -- Basic Idea

![Transformation vom *input space* in den *feature space*](../assets/feature_space.png)

- Transformiere Daten zunächst in den *feature space* und wende dort lineare SVM an

### Nonlinear SVM -- *feature space*

- $\Phi: \mathbb{R}^r \to \mathcal{H}$ als nicht-lineare Transformation
- alte Gleichung: $\min ||\beta||^2$ mit $y_i (\beta_0 + x_i^T \beta) \geq +1$
- neue Gleichung: $\min ||\beta||^2$ mit $y_i (\beta_0 + \Phi(x_i)^T \beta) \geq +1$
