import * as tf from '@tensorflow/tfjs';
import * as tfvis from '@tensorflow/tfjs-vis';


import factoringData from './datasets/factoring/factoring-data.json'
console.log('>>>>>', factoringData);

function onlyUnique(value, index, self) { 
  return self.indexOf(value) === index;
}

const uniqueCustomerIds = factoringData
  .map(record => record['customerID'])
  .filter(onlyUnique);
const uniqueDisputed = factoringData
  .map(record => record['Disputed'])
  .filter(onlyUnique);
const uniquePaperlessBill = factoringData
  .map(record => record['PaperlessBill'])
  .filter(onlyUnique);

const convertToFeatureVector = (
  features
) => {
  return [
    features['countryCode'],
    uniqueCustomerIds.indexOf(features['customerID']),
    (new Date(features['PaperlessDate']).getTime()),
    features['invoiceNumber'],
    (new Date(features['InvoiceDate']).getTime()),
    (new Date(features['DueDate']).getTime()),
    features['InvoiceAmount'],
    uniqueDisputed.indexOf(features['Disputed']),
    (new Date(features['SettledDate']).getTime()),
    uniquePaperlessBill.indexOf(features['PaperlessBill']),
    features['DaysToSettle'],
  ];
}

const values = factoringData.map(({
  DaysLate: daysLate,
  ...rest
}) => ({
    x: convertToFeatureVector(rest),
    y: daysLate
}));

console.log('values', values)

tfvis.visor();

tfvis.render.scatterplot(
    { name: 'features v DaysLate' },
    { values }, 
    {
        xLabel: 'features',
        yLabel: 'DaysLate',
        height: 300
    }
);

// Create a sequential model
const model = tf.sequential(); 
  
// Add a single input layer
model.add(tf.layers.dense({ units: 32, inputShape: [11], useBias: true, activation: 'relu'}));

// Add a few hidden layers
model.add(tf.layers.dense({ units: 128, activation: 'relu'}));
model.add(tf.layers.dense({ units: 1024, activation: 'relu'}));
model.add(tf.layers.dense({ units: 256, activation: 'relu'}));
model.add(tf.layers.dense({ units: 64, activation: 'relu'}));

// Add an output layer
model.add(tf.layers.dense({units: 1, useBias: true, activation: 'linear'}));

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
    const formattedData = data.map(({ DaysLate, ...input}) => {
      
      return ({
          input: convertToFeatureVector(input),
          label: DaysLate
      });
    });

    const inputs = formattedData.map(d => d.input);
    const labels = formattedData.map(d => d.label);

    const inputTensor = tf.tensor2d(inputs, [inputs.length, 11]);
    const labelTensor = tf.tensor2d(labels, [labels.length, 1]);

    //Step 3. Normalize the data to the range 0 - 1 using min-max scaling
    // const inputMax = inputTensor.max();
    // const inputMin = inputTensor.min();  
    // const labelMax = labelTensor.max();
    // const labelMin = labelTensor.min();

    // const normalizedInputs = inputTensor.sub(inputMin).div(inputMax.sub(inputMin));
    // const normalizedLabels = labelTensor.sub(labelMin).div(labelMax.sub(labelMin));

    return {
      inputs: inputTensor,
      labels: labelTensor,
      // Return the min/max bounds so we can use them later.
      // inputMax,
      // inputMin,
      // labelMax,
      // labelMin,
    }
  });  
}

// Convert the data to a form we can use for training.
const tensorData = convertToTensor(factoringData);
const {inputs, labels} = tensorData;
console.log('tensor created', inputs);

function trainModel(model, inputs, labels) {
  // Prepare the model for training.  
  model.compile({
    optimizer: tf.train.adam(),
    loss: tf.losses.meanSquaredError,
    metrics: ['mse'],
  });

  console.log('compiled model');
  
  const batchSize = 100;
  const epochs = 100;

  console.log({
    inputs,
    labels
  });
  
  return model.fit(inputs, labels, {
    batchSize,
    epochs,
    shuffle: true
  }).then((info) => console.log('training finished', info));
}
    
// Train the model  
trainModel(model, inputs, labels)
    .then(() => {
        console.log('Done Training');

        // test the model
        testModel(model, factoringData, tensorData)
    })

function testModel(model, inputData, normalizationData) {
  // const {inputMax, inputMin, labelMin, labelMax} = normalizationData;
  
  // // Generate predictions for a uniform range of numbers between 0 and 1;
  // // We un-normalize the data by doing the inverse of the min-max scaling 
  // // that we did earlier.
  // const [xs, preds] = tf.tidy(() => {
    
  //   const xs = tf.linspace(0, 11, 11);
  //   const preds = model.predict(xs.reshape([1, 11]));      
    
  //   const unNormXs = xs
  //     .mul(inputMax.sub(inputMin))
  //     .add(inputMin);
    
  //   const unNormPreds = preds
  //     .mul(labelMax.sub(labelMin))
  //     .add(labelMin);
    
  //   // Un-normalize the data
  //   return [unNormXs.dataSync(), unNormPreds.dataSync()];
  // });
  
 
  // const predictedPoints = Array.from(xs).map((val, i) => {
  //   console.log('predicting val', val, preds[i]);
  //   return {x: (val), y: preds[i]}
  // });
  
  const originalPoints = inputData.map(({ DaysLate, ...rest }) => {
    // const tf2 = tf;
    // console.log('-----rest', rest);
    // console.log('convertToFeatureVector(rest)', convertToFeatureVector(rest));
    // console.log('prediction',  model.predict(tf2.tensor(convertToFeatureVector(rest), [1, 11])).print());
   
    return ({
      x: convertToFeatureVector(rest),
      y: DaysLate,
      prediction: model.predict(tf.tensor(convertToFeatureVector(rest), [1, 11]))
    });
  });
  debugger;
  
  tfvis.render.scatterplot(
    {name: 'Model Predictions vs Original Data'}, 
    {values: [originalPoints.map(({ x, y }) => ({ x, y })), originalPoints.map(({ x, prediction }) => ({ x, y: prediction }))], series: ['original', 'predicted']}, 
    {
      xLabel: 'rest',
      yLabel: 'DaysLate',
      height: 300
    }
  );
}
