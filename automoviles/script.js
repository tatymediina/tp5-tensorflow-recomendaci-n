
      console.log('Hello TensorFlow');

      /**
       * Obtener y limpiar los datos
       */
      async function getData() {
        const carsDataResponse = await fetch('https://storage.googleapis.com/tfjs-tutorials/carsData.json');
        const carsData = await carsDataResponse.json();
        const cleaned = carsData.map(car => ({
          mpg: car.Miles_per_Gallon,
          horsepower: car.Horsepower,
        })).filter(car => (car.mpg != null && car.horsepower != null));
        return cleaned;
      }

      async function run() {
        // Obtener y visualizar los datos
        const data = await getData();
        const values = data.map(d => ({
          x: d.horsepower,
          y: d.mpg,
        }));

        tfvis.render.scatterplot(
          { name: 'Caballos de fuerza vs MPG' },
          { values },
          {
            xLabel: 'Caballos de fuerza',
            yLabel: 'Millas por galón (MPG)',
            height: 300
          }
        );

        const model = createModel();
        tfvis.show.modelSummary({ name: 'Resumen del modelo' }, model);

        const tensorData = convertToTensor(data);
        const { inputs, labels } = tensorData;

        await trainModel(model, inputs, labels);
        console.log('Entrenamiento finalizado');

        // Cuando actualices la página, deberías ver la comparación entre los datos reales y las predicciones
        testModel(model, data, tensorData);
      }

      document.addEventListener('DOMContentLoaded', run);

      function createModel() {
        const model = tf.sequential();
        model.add(tf.layers.dense({ inputShape: [1], units: 1, useBias: true }));
        model.add(tf.layers.dense({ units: 1, useBias: true }));
        return model;
      }

      function convertToTensor(data) {
        return tf.tidy(() => {
          tf.util.shuffle(data);

          const inputs = data.map(d => d.horsepower);
          const labels = data.map(d => d.mpg);

          const inputTensor = tf.tensor2d(inputs, [inputs.length, 1]);
          const labelTensor = tf.tensor2d(labels, [labels.length, 1]);

          const inputMax = inputTensor.max();
          const inputMin = inputTensor.min();
          const labelMax = labelTensor.max();
          const labelMin = labelTensor.min();

          const normalizedInputs = inputTensor.sub(inputMin).div(inputMax.sub(inputMin));
          const normalizedLabels = labelTensor.sub(labelMin).div(labelMax.sub(labelMin));

          return {
            inputs: normalizedInputs,
            labels: normalizedLabels,
            inputMax,
            inputMin,
            labelMax,
            labelMin,
          };
        });
      }

      async function trainModel(model, inputs, labels) {
        model.compile({
          optimizer: tf.train.adam(),
          loss: tf.losses.meanSquaredError,
          metrics: ['mse'],
        });

        const batchSize = 32;
        const epochs = 50;

        return await model.fit(inputs, labels, {
          batchSize,
          epochs,
          shuffle: true,
          callbacks: tfvis.show.fitCallbacks(
            { name: 'Rendimiento del entrenamiento' },
            ['loss', 'mse'],
            { height: 200, callbacks: ['onEpochEnd'] }
          )
        });
      }

      function testModel(model, inputData, normalizationData) {
        const { inputMax, inputMin, labelMin, labelMax } = normalizationData;

        const [xs, preds] = tf.tidy(() => {
          const xs = tf.linspace(0, 1, 100);
          const preds = model.predict(xs.reshape([100, 1]));

          const unNormXs = xs.mul(inputMax.sub(inputMin)).add(inputMin);
          const unNormPreds = preds.mul(labelMax.sub(labelMin)).add(labelMin);

          return [unNormXs.dataSync(), unNormPreds.dataSync()];
        });

        const predictedPoints = Array.from(xs).map((val, i) => {
          return { x: val, y: preds[i] };
        });

        const originalPoints = inputData.map(d => ({
          x: d.horsepower,
          y: d.mpg,
        }));

        tfvis.render.scatterplot(
          { name: 'Predicciones vs Datos Originales' },
          {
            values: [originalPoints, predictedPoints],
            series: ['original', 'predicciones'],
          },
          {
            xLabel: 'Caballos de fuerza',
            yLabel: 'Millas por galón (MPG)',
            height: 300,
          }
        );
      }
