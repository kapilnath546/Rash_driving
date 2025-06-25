import express from "express";
import cors from "cors";
import multer from "multer";
import path from "path";
import fs from "fs";
import { exec } from "child_process";

const app = express();
app.use(cors());

// Make sure uploads folder exists
const uploadFolder = './uploads';
if (!fs.existsSync(uploadFolder)) {
    fs.mkdirSync(uploadFolder);
}

const storage = multer.diskStorage({
    destination: function (req, file, cb) {
        cb(null, 'uploads/');
    },
    filename: function (req, file, cb) {
        cb(null, Date.now() + '-' + file.originalname);
    }
});

const upload = multer({ storage: storage });

// Upload Route
app.post('/upload', upload.single('file'), (req, res) => {
    console.log(req.file);

    const uploadedFilePath = req.file.path;
    const destinationPath = path.join(
        "C:/Users/kapil/Downloads/myvenv/rash/input",
        req.file.filename
    );

    fs.copyFile(uploadedFilePath, destinationPath, (err) => {
        if (err) {
            console.error("Error copying file:", err);
            return res.status(500).json({ error: "Error copying file" });
        }

        console.log("File copied to input folder successfully!");

        // 🛜 After file is copied, run the two commands
        exec(`start cmd.exe /k "cd C:\\Users\\kapil\\Downloads\\myvenv\\rash && py yolo.py"`, (error, stdout, stderr) => {
            if (error) {
                console.error(`Error executing commands: ${error.message}`);
                return;
            }
            if (stderr) {
                console.error(`stderr: ${stderr}`);
                return;
            }
            console.log(`stdout: ${stdout}`);
        });

        res.json({ message: "File uploaded, copied, and yolo.py started successfully!" });
    });
});

// Start Server
const PORT = 5000;
app.listen(PORT, () => {
    console.log(`Server running on http://localhost:${PORT}`);
});
