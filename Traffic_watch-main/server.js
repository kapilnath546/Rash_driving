import express from 'express';
import multer from 'multer';
import cors from 'cors';
import path from 'path';
import { fileURLToPath } from 'url';
import { dirname } from 'path';

// Fix __dirname because in ESModules, it doesn't exist
const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const app = express();
app.use(cors());

// Set up Multer storage
const storage = multer.diskStorage({
  destination: function (req, file, cb) {
    cb(null, __dirname + '/input/'); // absolute path to your input folder
  },
  filename: function (req, file, cb) {
    cb(null, Date.now() + '-' + file.originalname);
  }
});

const upload = multer({ storage: storage });

// API Endpoint
app.post('/upload', upload.single('file'), (req, res) => {
  console.log(req.file);
  res.json({ message: 'File uploaded successfully!' });
});

app.listen(5000, () => {
  console.log('Server running at http://localhost:5000');
});
