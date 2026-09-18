let models;

const form = document.querySelector("#prediction-form");
const freshness = document.querySelector("#freshness");
const freshnessValue = document.querySelector("#freshness-value");
const predictButton = document.querySelector("#predict-button");
const status = document.querySelector("#model-status");
const result = document.querySelector("#result");

freshness.addEventListener("input", () => {
  freshnessValue.textContent = `${freshness.value} / 10`;
});

function makeFeatures() {
  const fruit = document.querySelector("#fruit").value;
  const season = document.querySelector("#season").value;
  const origin = document.querySelector("#origin").value;

  return [
    Number(freshness.value),
    Number(document.querySelector("#weight").value) / 1000,
    Number(fruit === "Banana"),
    Number(fruit === "Grapes"),
    Number(fruit === "Mango"),
    Number(fruit === "Orange"),
    Number(season === "Spring"),
    Number(season === "Summer"),
    Number(season === "Winter"),
    Number(origin === "Local"),
  ];
}

function predictLinear(features) {
  return models.linear.intercept + features.reduce(
    (sum, value, index) => sum + value * models.linear.coefficients[index], 0
  );
}

function predictTree(tree, features) {
  let node = 0;
  while (tree.childrenLeft[node] !== -1) {
    // scikit-learn evaluates tree inputs as float32 values.
    node = Math.fround(features[tree.feature[node]]) <= tree.threshold[node]
      ? tree.childrenLeft[node]
      : tree.childrenRight[node];
  }
  return tree.value[node];
}

function predictForest(features) {
  const total = models.forest.trees.reduce((sum, tree) => sum + predictTree(tree, features), 0);
  return total / models.forest.trees.length;
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  const features = makeFeatures();
  const model = new FormData(form).get("model");
  const prediction = model === "linear" ? predictLinear(features) : predictForest(features);
  const fruit = document.querySelector("#fruit").value;

  document.querySelector("#price").textContent = `$${Math.max(0, prediction).toFixed(2)}`;
  document.querySelector("#result-details").textContent = `${fruit} · ${model === "linear" ? "Linear Regression" : "Random Forest"}`;
  result.hidden = false;
});

fetch("model.json")
  .then((response) => {
    if (!response.ok) throw new Error("Model file could not be loaded.");
    return response.json();
  })
  .then((data) => {
    models = data;
    status.textContent = "Models ready";
    status.classList.add("ready");
    predictButton.disabled = false;
  })
  .catch((error) => {
    status.textContent = "Loading failed";
    console.error(error);
  });
