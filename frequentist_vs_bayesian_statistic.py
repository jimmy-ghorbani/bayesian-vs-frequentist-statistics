
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import pymc as pm
import seaborn as sns
import arviz as az
from scipy import stats
from sklearn.model_selection import train_test_split
from statsmodels.stats.proportion import proportion_confint
from sklearn.linear_model import LinearRegression




data = yf.download('SPY GLD USO SLV',period='1y')
data.head()
data = data.loc[:,'Close']
returns = np.log(data / data.shift(1))
returns = returns.dropna()

returns.corr()['GLD'].sort_values()

returns['Gold Binary'] = (returns['GLD'] > 0).astype(int)
returns

n_days = [10, 25, 50, 120,200, len(returns)]
outcomes = returns['Gold Binary'].sample(n_days[-1], random_state=42)
p = np.linspace(0, 1, 100)

# Uniform prior
a = b = 1

# Create 2 rows of subplots
fig, axes = plt.subplots(3, 2, figsize=(10, 8))
axes = axes.flatten()  # Flatten to use single loop

for i, days in enumerate(n_days):
    up = outcomes.iloc[:days].sum()
    down = days - up
    α_post = a + up
    β_post = b + down
    update = stats.beta.pdf(p, α_post, β_post)
    mean = α_post / (α_post + β_post)

    axes[i].plot(p, update, label=f"{days} days\nUp: {int(up)} | Down: {int(down)}")
    axes[i].axvline(mean, color='red', linestyle='--', label=f"Mean: {mean:.2f}")
    axes[i].set_title(f"Posterior after {days} days")
    axes[i].set_xlabel("p (Probability of Win)")
    axes[i].set_ylabel("Density")
    axes[i].legend()

plt.tight_layout()
plt.show()

n_days = [10, 25, 50, 120,200, 249]
alpha =  0.05
total_days = len(returns)
outcomes = returns['Gold Binary'].sample(n_days[-1], random_state=42)

fig, axes = plt.subplots(3, 2, figsize=(10, 8))
axes = axes.flatten()

for i, days in enumerate(n_days):
    positive_days = outcomes.iloc[:days].sum()
    MLE = positive_days / days
    SD = np.sqrt(MLE * (1 - MLE))
    SE = SD / np.sqrt(days)
    lower, upper = proportion_confint(positive_days, days, alpha=alpha, method='normal')

    # Plot vertical lines on subplot
    axes[i].set_xlim(0, 1)
    axes[i].axvline(MLE, color='blue', linestyle='--', label=f'MLE = {MLE:.4f}')
    axes[i].axvline(lower, color='green', linestyle=':', label=f'Lower 95% CI = {lower:.4f}')
    axes[i].axvline(upper, color='green', linestyle=':', label=f'Upper 95% CI = {upper:.4f}')

    # Subplot titles and labels
    axes[i].set_title(f"{days} Days of Data")
    axes[i].set_xlabel('Probability')
    axes[i].set_ylabel('Density')
    axes[i].legend()

plt.tight_layout()
plt.show()

"""# implementing a simple machine learning model with bayesian and frequentist approaches

In practice, calculating the exact posterior distribution is computationally intractable for continuous values and so we turn to sampling methods such as Markov Chain Monte Carlo (MCMC) to draw samples from the posterior in order to approximate the posterior. Monte Carlo refers to the general technique of drawing random samples, and Markov Chain means the next sample drawn is based only on the previous sample value. The concept is that as we draw more samples, the approximation of the posterior will eventually converge on the true posterior distribution for the model parameters.

The best library for probabilistic programming and Bayesian Inference in Python is currently PyMC. It includes numerous utilities for constructing Bayesian Models and using MCMC methods to infer the model parameters.

There are only two steps we need to do to perform Bayesian Linear Regression with this module:

1-Build a formula relating the features to the target and decide on a prior distribution for the data likelihood</br>
2-Sample from the parameter posterior distribution using MCMC
"""

def test_model(trace, test_observation, actual):
    #print('Test Observation:')
    #print(test_observation)

    # See all variable names
    print("Available variables:", list(trace.posterior.data_vars))

    # Extract means
    var_means = {}
    for var in trace.posterior.data_vars:
        mean_val = trace.posterior[var].mean(dim=("chain", "draw")).values
        var_means[var] = mean_val

    intercept_name = [var for var in var_means if "β0" in var or "intercept" in var.lower()][0]
    theta_name = [var for var in var_means if "θ" in var or "theta" in var.lower()][0]
    sigma_name = [var for var in var_means if "σ" in var or "sigma" in var.lower()][0]

    intercept = var_means[intercept_name]
    coefs = var_means[theta_name]
    sigma = var_means[sigma_name]

    test_features = test_observation.values

    mean_loc = intercept + np.dot(test_features, coefs)

    # Generate posterior predictive samples
    estimates = np.random.logistic(loc=mean_loc, scale=sigma, size=1000)


    # Extract the actual value using integer-based indexing
    index = test_observation.name
    actual_value = actual.loc[index]  # Use integer-based indexing for the correct observation

    # Calculate the residuals (error terms)
    residuals = estimates - actual_value

    # Plotting the posterior predictive distribution
    plt.figure(figsize=(8, 6))
    sns.histplot(estimates, kde=True, bins=20, color="skyblue", edgecolor="k", stat="density")

    # Mean of the estimates
    plt.axvline(x=mean_loc, color="orange", linestyle="-", label="Mean Estimate", linewidth=2.5)

    # Plot actual value (true grade)
    plt.axvline(x=actual_value, color="red", linestyle="--", label="True Value (Actual)", linewidth=2.5)
    plt.legend()

    # Print summary statistics

    print(f"5% Estimate: {np.percentile(estimates, 5):.4f}")
    print(f"95% Estimate: {np.percentile(estimates, 95):.4f}")
    print(f"Mean Residual: {np.mean(residuals):.4f}")
    print(f"Residual Std Dev: {np.std(residuals):.4f}")


#fit distribution to the returns and select the one with highest likelihood
def fit_dist(selected_stocks, returns):
    marginals_df = pd.DataFrame(index=selected_stocks, columns=['Distribution', 'AIC', 'BIC', 'KS_pvalue'])
    for stock in selected_stocks:
        data = returns[stock]
        dists = ['Normal', "t-Student", 'Logistic', 'Exponential']
        best_aic = np.inf
        for dist,name in zip([stats.norm, stats.t, stats.genlogistic, stats.genextreme, stats.expon, stats.gamma], dists):
            params = dist.fit(data)
            dist_fit = dist(*params)
            log_like = np.log(dist_fit.pdf(data)).sum()
            aic = 2*len(params) - 2 * log_like
            if aic<best_aic:
                best_dist = name
                best_aic = aic
                best_bic = len(params) * np.log(len(data)) - 2 * log_like
                ks_pval = stats.kstest(data, dist_fit.cdf, N=100)[1]
        marginals_df.loc[stock] = [best_dist, best_aic, best_bic, ks_pval]
    return marginals_df

selected_stocks = returns.columns[:-1].tolist()

fit_dist(selected_stocks, returns)

#splitting the data into train and test sets
labels = returns['GLD']
df = returns[['SPY', 'USO', 'SLV']]
X_train, X_test, y_train, y_test = train_test_split(df, labels,
                                                    test_size = 0.20,
                                                    random_state=42)
X_train.head()

"""we assume that the relationship between the predictors and the response follows alinear model with a logistic likelihood. We first set priors for the model parameters: a Normal prior for the intercept (β₀), independent Normal priors for the regression coefficients (θ), and a Half-Normal prior for the scale parameter σ to ensure it remains positive. The expected value of the response (μ) is modeled as a linear combination of the predictors and coefficients plus the intercept. Given μ, the likelihood of the observed data (y_obs) is specified using a Logistic distribution, where σ controls the spread of the outcomes around μ. Finally, we perform posterior sampling using Markov Chain Monte Carlo (MCMC), generating 1000 samples after 1000 tuning steps, and store the results as an InferenceData object to facilitate posterior analysis.


"""

with pm.Model() as model:
    # Priors for intercept and coefficients
    β0 = pm.Normal("β0", mu=0, sigma=10)
    θ = pm.Normal("θ", mu=0, sigma=10, shape=X_train.shape[1])
    σ = pm.HalfNormal("σ", sigma=1)

    # Expected value of outcome
    mu = β0 + pm.math.dot(X_train, θ)

    # Likelihood
    y_obs = pm.Logistic("y_obs", mu=mu, s=σ, observed=y_train)

    # Sampling
    trace = pm.sample(1000, tune=1000, return_inferencedata=True)

# Plot posterior distributions
az.plot_posterior(
    trace,
    figsize=(12, 8),
    kind='kde',
    point_estimate='mean',
    hdi_prob=0.95,
    textsize=12,
    round_to=3,
)

plt.tight_layout()
plt.show()

az.summary(trace)

print(trace.posterior.dims)



test_model(trace, X_test.iloc[34],y_test)

# Create and fit the model
lr = LinearRegression()
model = lr.fit(X_train, y_train)

# Print coefficients and intercept
print("Intercept (β0):", lr.intercept_)
print("Coefficients (β):", lr.coef_)

y_pred = model.predict(X_test)

# Calculate residuals (errors)
residuals = y_test.values - y_pred
print(f"Mean Residual: {residuals.mean():.4f}")
print(f"Residual Std Dev: {residuals.std():.4f}")
print(f"5% Residual: {np.percentile(residuals, 5):.4f}")
print(f"95% Residual: {np.percentile(residuals, 95):.4f}")

"""# **Conclusion**

Bayesian and Frequentist approaches offer two different philosophies for statistical modeling. The Frequentist view treats parameters as fixed but unknown, and focuses on estimating them purely from data without incorporating prior beliefs; it is often simpler, computationally faster, and works well when there is abundant data. However, it struggles when data is scarce or when uncertainty quantification is important. Bayesian methods, by contrast, treat parameters as random variables and update beliefs about them using observed data and prior knowledge. This allows Bayesian models to handle small datasets more robustly, naturally provide full distributions for forecasts (not just point estimates), and incorporate expert information. The downside is that Bayesian methods can be computationally intensive and sensitive to the choice of prior when data is very limited. In practice, Frequentist methods are often used for quick, large-sample problems, while Bayesian approaches are favored when modeling uncertainty is crucial, data is limited, or domain knowledge should be built into the model.
"""