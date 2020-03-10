import * as tf from '@tensorflow/tfjs';
import * as tfvis from '@tensorflow/tfjs-vis';


import factoringData from './datasets/factoring/factoring-data.json'
console.log('>>>>>', factoringData);

const values = factoringData.map(({
    InvoiceAmount: invoiceAmount,
    DaysLate: daysLate
}) => ({
    x: invoiceAmount,
    y: daysLate
}))

tfvis.visor();

tfvis.render.scatterplot(
    { name: 'InvoiceAmount v DaysLate' },
    { values }, 
    {
        xLabel: 'InvoiceAmount',
        yLabel: 'DaysLate',
        height: 300
    }
);

// Create a sequential model
const model = tf.sequential(); 
  
// Add a single hidden layer
model.add(tf.layers.dense({inputShape: [1], units: 1, useBias: true}));

// Add an output layer
model.add(tf.layers.dense({units: 1, useBias: true}));

tfvis.show.modelSummary({name: 'Model Summary'}, model);

/**
 * Convert the input data to tensors that we can use for machine 
 * learning. We will also do the important best practices of _shuffling_
 * the data and _normalizing_ the data
 * MPG on the y-axis.
 */
function convertToTensor(data) {
    // Wrapping these calculations in a tidy will dispose any 
    // intermediate tensors.
    
    return tf.tidy(() => {
      // Step 1. Shuffle the data    
      tf.util.shuffle(data);
  
      // Step 2. Convert data to Tensor
      const formattedData = data.map(({ DaysLate, ...input}) => ({
          input,
          label: DaysLate
      }));

      const inputs = formattedData.map(d => Object.keys(d.input).map(prop => ([prop, `${d.input[prop]}`.split('').map(Number)])))
      const labels = formattedData.map(d => d.label);
  
      const inputTensor = tf.tensor3d(inputs, [inputs.length, inputs[0].length, 2]);
      const labelTensor = tf.tensor2d(labels, [labels.length, 1]);
  
      //Step 3. Normalize the data to the range 0 - 1 using min-max scaling
      const inputMax = inputTensor.max();
      const inputMin = inputTensor.min();  
      const labelMax = labelTensor.max();
      const labelMin = labelTensor.min();
  
      const normalizedInputs = inputTensor.sub(inputMin).div(inputMax.sub(inputMin));
      const normalizedLabels = labelTensor.sub(labelMin).div(labelMax.sub(labelMin));
  
      return {
        inputs: normalizedInputs,
        labels: normalizedLabels,
        // Return the min/max bounds so we can use them later.
        inputMax,
        inputMin,
        labelMax,
        labelMin,
      }
    });  
  }

  function trainModel(model, inputs, labels) {
    // Prepare the model for training.  
    model.compile({
      optimizer: tf.train.adam(),
      loss: tf.losses.meanSquaredError,
      metrics: ['mse'],
    });
    
    const batchSize = 100;
    const epochs = 100;
    
    return model.fit(inputs, labels, {
      batchSize,
      epochs,
      shuffle: true,
      callbacks: tfvis.show.fitCallbacks(
        { name: 'Training Performance' },
        ['loss', 'mse'], 
        { height: 200, callbacks: ['onEpochEnd'] }
      )
    });
}

// Convert the data to a form we can use for training.
const tensorData = convertToTensor(factoringData);
const {inputs, labels} = tensorData;
    
// Train the model  
trainModel(model, inputs, labels)
    .then(() => {
        console.log('Done Training');

        // test the model
        testModel(model, factoringData, tensorData)
    });

function testModel(model, inputData, normalizationData) {
  const {inputMax, inputMin, labelMin, labelMax} = normalizationData;  
  
  // Generate predictions for a uniform range of numbers between 0 and 1;
  // We un-normalize the data by doing the inverse of the min-max scaling 
  // that we did earlier.
  const [xs, preds] = tf.tidy(() => {
    
    const xs = tf.linspace(0, 1, 100);      
    const preds = model.predict(xs.reshape([100, 1]));      
    
    const unNormXs = xs
      .mul(inputMax.sub(inputMin))
      .add(inputMin);
    
    const unNormPreds = preds
      .mul(labelMax.sub(labelMin))
      .add(labelMin);
    
    // Un-normalize the data
    return [unNormXs.dataSync(), unNormPreds.dataSync()];
  });
  
 
  const predictedPoints = Array.from(xs).map((val, i) => {
    return {x: val, y: preds[i]}
  });
  
  const originalPoints = inputData.map(d => ({
    x: d.InvoiceAmount, y: d.DaysLate,
  }));
  
  
  tfvis.render.scatterplot(
    {name: 'Model Predictions vs Original Data'}, 
    {values: [originalPoints, predictedPoints], series: ['original', 'predicted']}, 
    {
      xLabel: 'InvoiceAmount',
      yLabel: 'DaysLate',
      height: 300
    }
  );
}

// testModel(model, factoringData, tensorData);
