const express = require('express');
const fs = require('fs');
const fsPromises = require('fs').promises;
const path = require('path');
const yaml = require('js-yaml');
const { Image, createCanvas } = require('canvas');
const NodeCache = require('node-cache');
const cors = require('cors');
const { spawn } = require('child_process');

const app = express();
app.use(cors());
app.use(express.json({ limit: '50mb' }));

const cache = new NodeCache({ stdTTL: 3600 });

const loadConfig = async (yamlPath) => {
  try {
    const file = await fsPromises.readFile(yamlPath, 'utf8');
    return yaml.load(file);
  } catch (error) {
    throw new Error(`Failed to load YAML config: ${error}`);
  }
};

const getImageFiles = async (basePath, relativePath) => {
  const fullPath = path.join(basePath, relativePath);
  try {
    if (!fs.existsSync(fullPath)) return [];
    const files = await fsPromises.readdir(fullPath);
    return files
      .filter(f => /\.(jpg|jpeg|png|bmp|gif|tiff|webp)$/i.test(f))
      .map(f => path.join(fullPath, f))
      .sort();
  } catch (error) {
    return [];
  }
};

const getLabelPath = (imagePath) => {
  const baseDir = path.dirname(imagePath);
  const imgName = path.basename(imagePath);
  const labelName = `${path.parse(imgName).name}.txt`;
  let labelPath = path.join(baseDir.replace('images', 'labels'), labelName);
  if (fs.existsSync(path.dirname(labelPath))) return labelPath;
  labelPath = path.join(path.dirname(baseDir), 'labels', path.basename(baseDir), labelName);
  if (fs.existsSync(path.dirname(labelPath))) return labelPath;
  return path.join(baseDir, labelName);
};

const parseYoloLabels = async (labelPath) => {
  if (!fs.existsSync(labelPath)) return [];
  try {
    const content = await fsPromises.readFile(labelPath, 'utf8');
    return content.split('\n')
      .filter(line => line.trim())
      .map(line => {
        const parts = line.split(' ').map(Number);
        if (parts.length === 5) return parts;
        return null;
      })
      .filter(line => line);
  } catch (error) {
    return [];
  }
};

const yoloToAbs = (yoloCoords, imgWidth, imgHeight) => {
  const [classId, xCenter, yCenter, width, height] = yoloCoords;
  const absWidth = width * imgWidth;
  const absHeight = height * imgHeight;
  const xMin = (xCenter * imgWidth) - (absWidth / 2);
  const yMin = (yCenter * imgHeight) - (absHeight / 2);
  return [classId, xMin, yMin, xMin + absWidth, yMin + absHeight];
};

const absToYolo = (absCoords, imgWidth, imgHeight) => {
  const [classId, xMin, yMin, xMax, yMax] = absCoords;
  const xCenter = ((xMin + xMax) / 2) / imgWidth;
  const yCenter = ((yMin + yMax) / 2) / imgHeight;
  const width = (xMax - xMin) / imgWidth;
  const height = (yMax - yMin) / imgHeight;
  return [classId, xCenter, yCenter, width, height];
};

app.post('/delete-images', async (req, res) => {
    const { imagePaths } = req.body;
    if (!imagePaths || !Array.isArray(imagePaths)) {
        return res.status(400).json({ error: 'imagePaths must be an array.' });
    }

    const deletionPromises = imagePaths.map(async (imagePath) => {
        const labelPath = getLabelPath(imagePath);
        try {
            if (fs.existsSync(imagePath)) await fsPromises.unlink(imagePath);
            if (fs.existsSync(labelPath)) await fsPromises.unlink(labelPath);
            return { path: imagePath, status: 'success' };
        } catch (error) {
            return { path: imagePath, status: 'error', reason: error.message };
        }
    });

    try {
        const results = await Promise.all(deletionPromises);
        res.json({ success: true, results });
    } catch (error) {
        res.status(500).json({ error: `An unexpected error occurred: ${error.message}` });
    }
});


app.get('/', (req, res) => {
  res.json({ message: 'Welcome to the YOLOv8 Dataset Manager Backend!' });
});

app.get('/load-data', async (req, res) => {
  try {
    const { yamlPath } = req.query;
    const config = await loadConfig(yamlPath);
    const datasetRoot = path.isAbsolute(config.path) ? config.path : path.join(path.dirname(yamlPath), config.path);
    const imageSets = {
      train: await getImageFiles(datasetRoot, config.train || ''),
      valid: await getImageFiles(datasetRoot, config.val || ''),
      test: await getImageFiles(datasetRoot, config.test || '')
    };
    res.json({ imageSets, classNames: config.names || {} });
  } catch (error) {
    res.status(500).json({ error: `Server error: ${error.message}` });
  }
});

app.post('/load-images', async (req, res) => {
    try {
        const { paths, viewMode } = req.body;
        const images = [];
        for (const imgPath of paths) {
            const labels = await parseYoloLabels(getLabelPath(imgPath));
            if (viewMode === 'Background' && labels.length > 0) {
                continue;
            }

            const cacheKey = `${imgPath}_${viewMode}`;
            let imgData = cache.get(cacheKey);

            if (!imgData) {
                const img = new Image();
                img.src = await fsPromises.readFile(imgPath);
                const canvas = createCanvas(img.width, img.height);
                const ctx = canvas.getContext('2d');
                ctx.drawImage(img, 0, 0);

                if (viewMode === 'Annotated') {
                    labels.forEach(label => {
                        const [classId, xMin, yMin, xMax, yMax] = yoloToAbs(label, img.width, img.height);
                        ctx.strokeStyle = '#f472b6';
                        ctx.lineWidth = 2;
                        ctx.strokeRect(xMin, yMin, xMax - xMin, yMax - yMin);
                    });
                }
                imgData = { path: imgPath, base64: canvas.toDataURL(), numLabels: labels.length };
                cache.set(cacheKey, imgData);
            }
            images.push(imgData);
        }
        res.json(images);
    } catch (error) {
        res.status(500).json({ error: `Server error: ${error.message}` });
    }
});

app.get('/annotations', async (req, res) => {
    const { imagePath } = req.query;
    const labels = await parseYoloLabels(getLabelPath(imagePath));
    const img = new Image();
    img.src = await fsPromises.readFile(imagePath);
    const annotations = labels.map(label => {
        const [classId, xMin, yMin, xMax, yMax] = yoloToAbs(label, img.width, img.height);
        return { classId, xMin, yMin, xMax, yMax };
    });
    res.json(annotations);
});

app.post('/save-annotations', async (req, res) => {
    const { imagePath, annotations } = req.body;
    const labelPath = getLabelPath(imagePath);
    const img = new Image();
    img.src = await fsPromises.readFile(imagePath);
    const yoloAnnotations = annotations.map(ann => absToYolo(
        [ann.classId, ann.xMin, ann.yMin, ann.xMax, ann.yMax],
        img.width,
        img.height
    ));
    await fsPromises.mkdir(path.dirname(labelPath), { recursive: true });
    await fsPromises.writeFile(labelPath, yoloAnnotations.map(ann => ann.join(' ')).join('\n'));
    cache.del(`${imagePath}_Annotated`);
    res.json({ success: true });
});

app.post('/run-inference', (req, res) => {
    const { modelPath, imagePath } = req.body;
    const pythonExecutable = '../venv/bin/python';
    const pythonProcess = spawn(pythonExecutable, ['run_yolo.py', modelPath, imagePath]);
    let rawData = '';
    pythonProcess.stdout.on('data', (data) => {
        rawData += data.toString();
    });
    let errorData = '';
    pythonProcess.stderr.on('data', (data) => {
        errorData += data.toString();
    });
    pythonProcess.on('close', async (code) => {
        if (code !== 0) {
            return res.status(500).json({ error: 'Failed to run inference.', details: errorData });
        }
        try {
            const results = JSON.parse(rawData);
            if (results.error) {
                return res.status(500).json({ error: results.error });
            }
            const img = new Image();
            img.src = await fsPromises.readFile(imagePath);
            const canvas = createCanvas(img.width, img.height);
            const ctx = canvas.getContext('2d');
            ctx.drawImage(img, 0, 0);
            results.boxes.forEach((box, index) => {
                const [xMin, yMin, xMax, yMax] = box;
                const classId = results.class_ids[index];
                ctx.strokeStyle = '#FF69B4';
                ctx.lineWidth = 2;
                ctx.strokeRect(xMin, yMin, xMax - xMin, yMax - yMin);
            });
            res.json({ base64: canvas.toDataURL() });
        } catch (error) {
            res.status(500).json({ error: error.message });
        }
    });
});

app.get('/analytics', async (req, res) => {
    const { yamlPath } = req.query;
    const cacheKey = `analytics_${yamlPath}`;
    let analytics = cache.get(cacheKey);

    if (!analytics) {
        try {
            const config = await loadConfig(yamlPath);
            const datasetRoot = path.isAbsolute(config.path) ? config.path : path.join(path.dirname(yamlPath), config.path);
            
            const calculateSplitAnalytics = async (relativePath) => {
                const imagePaths = await getImageFiles(datasetRoot, relativePath);
                let singleBox = 0;
                let multiBox = 0;
                let zeroBox = 0;
                for (const imgPath of imagePaths) {
                    const labels = await parseYoloLabels(getLabelPath(imgPath));
                    if (labels.length === 0) zeroBox++;
                    else if (labels.length === 1) singleBox++;
                    else multiBox++;
                }
                return { total: imagePaths.length, singleBox, multiBox, zeroBox };
            };

            analytics = {
                train: await calculateSplitAnalytics(config.train || ''),
                valid: await calculateSplitAnalytics(config.val || ''),
                test: await calculateSplitAnalytics(config.test || '')
            };
            cache.set(cacheKey, analytics);
        } catch (error) {
            return res.status(500).json({ error: `Analytics error: ${error.message}` });
        }
    }
    res.json(analytics);
});

app.listen(3000, () => console.log('Server running on port 3000'))