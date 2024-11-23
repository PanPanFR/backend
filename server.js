const Hapi = require('@hapi/hapi');
const { Storage } = require('@google-cloud/storage');
const { Firestore } = require('@google-cloud/firestore');
const tf = require('@tensorflow/tfjs-node');
const { v4: uuidv4 } = require('uuid');

// Konfigurasi Google Cloud
const storage = new Storage();
const firestore = new Firestore();
const bucketName = 'ml-model-pandu';

// Fungsi untuk memuat model TensorFlow.js dari Cloud Storage
let model;
const loadModel = async () => {
    const bucket = storage.bucket(bucketName);
    const file = bucket.file('ml-model-pandu/model.json');
    const [fileData] = await file.download();
    model = await tf.loadLayersModel(tf.io.bytesLoader(fileData));
};

// Endpoint untuk prediksi
const init = async () => {
    const server = Hapi.server({
        port: 8000,
        host: '0.0.0.0',
        routes: { cors: true },
    });

    server.route({
        method: 'POST',
        path: '/predict',
        options: {
            payload: {
                maxBytes: 1000000,
                output: 'stream',
                parse: true,
                allow: 'multipart/form-data',
            },
        },
        handler: async (request, h) => {
            try {
                const file = request.payload.image;
                if (!file || file.hapi.headers['content-type'].startsWith('image') === false) {
                    return h.response({ status: 'fail', message: 'File harus berupa gambar.' }).code(400);
                }

                const imageBuffer = await file._data;
                const tensor = tf.node.decodeImage(imageBuffer, 3).resizeBilinear([224, 224]).expandDims(0);
                const prediction = model.predict(tensor).dataSync();
                const result = prediction[0] > 0.5 ? 'Cancer' : 'Non-cancer';
                const suggestion = result === 'Cancer' ? 'Segera periksa ke dokter!' : 'Penyakit kanker tidak terdeteksi.';
                const id = uuidv4();
                const createdAt = new Date().toISOString();

                // Simpan hasil ke Firestore
                const predictionsRef = firestore.collection('predictions');
                await predictionsRef.doc(id).set({ id, result, suggestion, createdAt });

                return h.response({
                    status: 'success',
                    message: 'Model is predicted successfully',
                    data: { id, result, suggestion, createdAt },
                }).code(200);
            } catch (err) {
                console.error(err);
                return h.response({ status: 'fail', message: 'Terjadi kesalahan dalam melakukan prediksi' }).code(400);
            }
        },
    });

    await loadModel();
    await server.start();
    console.log(`Server running at: ${server.info.uri}`);
};

init();
