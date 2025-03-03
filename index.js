let chart;       // Chart.js instance for loss graph
let chart3d;     // ECharts instance for 3D visualization
let activationFn = "SIGMOID";
let layers = [2, 4, 1]; // Default architecture

document.getElementById("activation").addEventListener("change", function(event) {
  activationFn = event.target.value;
});

// Parse architecture string (e.g., "2,4,1") into an integer array.
function parseArchitecture() {
  let archStr = document.getElementById("arch").value;
  return archStr.split(",").map(s => parseInt(s.trim()));
}

// Helper: compute target function value for (x, y) based on selected target.
function targetFunctionValue(x, y, target) {
  switch (target) {
    case "FANCY_SINE":
      return (Math.sin(4 * Math.PI * x) * Math.cos(4 * Math.PI * y) + 1) / 2;
    case "SIMPLE_SINE":
      return (Math.sin(2 * Math.PI * x) + 1) / 2;
    case "SPIRAL":
      let theta = Math.atan2(y - 0.5, x - 0.5);
      return (Math.sin(5 * theta) + 1) / 2;
    case "XOR":
    default:
      return (x > 0.5 ^ y > 0.5) ? 1 : 0;
  }
}

// Generate synthetic dataset based on target function.
function generateDataset(target, numSamples = 100) {
  let X = [];
  let Y = [];
  for (let i = 0; i < numSamples; i++) {
    let x = Math.random();
    let y = Math.random();
    X.push([x, y]);
    Y.push([targetFunctionValue(x, y, target)]);
  }
  return { X, Y };
}

let nnGlobal = null;

async function trainModel() {
  let lr = parseFloat(document.getElementById("lr").value);
  let momentum = parseFloat(document.getElementById("momentum").value);
  let epochs = parseInt(document.getElementById("epochs").value);
  let batchSize = parseInt(document.getElementById("batchSize").value);
  let arch = parseArchitecture();
  let activation = activationFn;
  let targetFunc = document.getElementById("targetFunction").value;
  
  let Module = await import("./mlp.js");
  let MLP = Module.MLP;
  let ActFunc = Module.ActivationFunction;
  
  nnGlobal = new MLP(arch, lr, momentum, ActFunc[activation]);
  
  // Check for dataset upload; if none, generate synthetic data.
  let datasetFile = document.getElementById("datasetFile").files[0];
  let X, Y;
  if (datasetFile) {
    let data = await datasetFile.text();
    let lines = data.trim().split("\n");
    X = [];
    Y = [];
    for (let line of lines) {
      let values = line.split(",").map(Number);
      X.push(values.slice(0, values.length - 1));
      Y.push([values[values.length - 1]]);
    }
  } else {
    ({ X, Y } = generateDataset(targetFunc, 200));
  }
  
  let lossValues = nnGlobal.trainAndReturnLoss(X, Y, epochs, batchSize);
  
  document.getElementById("output").innerText =
    "Trained Weights: " + JSON.stringify(nnGlobal.getWeights());
  
  drawLossChart(lossValues);
  update3DSurface(targetFunc);
}

function drawLossChart(lossValues) {
  let ctx = document.getElementById("chart").getContext("2d");
  let epochs = lossValues.length;
  if (!chart) {
    chart = new Chart(ctx, {
      type: "line",
      data: {
        labels: [...Array(epochs).keys()].map(i => i + 1),
        datasets: [{
          label: "Loss",
          data: lossValues,
          borderColor: "#4285F4",
          fill: false,
          tension: 0.2
        }]
      },
      options: {
        scales: {
          x: { title: { display: true, text: "Epoch" } },
          y: {
            type: 'logarithmic',
            title: { display: true, text: "Loss (Log Scale)" },
            min: 0.0001
          }
        }
      }
    });
  } else {
    chart.data.labels = [...Array(epochs).keys()].map(i => i + 1);
    chart.data.datasets[0].data = lossValues;
    chart.update();
  }
}

function update3DSurface(target) {
  let gridSize = 30;
  let targetData = [];
  let modelData = [];
  for (let i = 0; i < gridSize; i++) {
    for (let j = 0; j < gridSize; j++) {
      let x = i / (gridSize - 1);
      let y = j / (gridSize - 1);
      let tVal = targetFunctionValue(x, y, target);
      targetData.push([x, y, tVal]);
      let pVal = nnGlobal ? nnGlobal.predict([x, y])[0] : 0;
      modelData.push([x, y, pVal]);
    }
  }
  
  if (!chart3d) {
    chart3d = echarts.init(document.getElementById("chart3d"));
  }
  
  let option = {
    tooltip: {},
    xAxis3D: { type: 'value', name: 'Input X' },
    yAxis3D: { type: 'value', name: 'Input Y' },
    zAxis3D: { type: 'value', name: 'Output' },
    grid3D: { viewControl: { projection: 'perspective' } },
    series: [
      {
        name: 'Target Function',
        type: 'surface',
        data: targetData,
        shading: 'lambert',
        itemStyle: { opacity: 0.7, color: '#34A853' },
        label: { show: false }
      },
      {
        name: 'Model Prediction',
        type: 'surface',
        data: modelData,
        shading: 'lambert',
        itemStyle: { opacity: 0.7, color: '#4285F4' },
        label: { show: false }
      }
    ]
  };
  
  chart3d.setOption(option);
}

function saveModel() {
  if (!nnGlobal) {
    alert("Train a model first!");
    return;
  }
  let weights = nnGlobal.getWeights();
  let blob = new Blob([JSON.stringify(weights)], { type: "application/json" });
  let a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "model_weights.json";
  a.click();
}

async function loadModel() {
  let file = document.getElementById("loadModelFile").files[0];
  if (!file) {
    alert("Please select a model file to load.");
    return;
  }
  let text = await file.text();
  let loadedWeights = JSON.parse(text);
  if (nnGlobal) {
    nnGlobal.setWeights(loadedWeights);
    document.getElementById("output").innerText =
      "Loaded Weights: " + JSON.stringify(nnGlobal.getWeights());
    update3DSurface(document.getElementById("targetFunction").value);
  } else {
    alert("No model instance exists. Train a model first.");
  }
}
