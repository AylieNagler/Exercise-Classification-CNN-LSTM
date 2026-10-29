# Exercise-Classification-CNN-LSTM
Project: AI Personal Trainer – Real-Time Workout Recognition with CNN-LSTM

Group Members:

    Aylie Lapierre Nagler, Arsi Kharghani, Idongesit Orok, Tom Yu, Kim Nguyen, Fahad Pasha


Objective:
To build a model that can watch a short video of someone working out and tell which exercise they are doing. We want it to be accurate but also fast enough to run live on a laptop webcam, so it can work like a personal trainer that keeps track of your workout for you.

Data:
We are using the UCF101 dataset (Soomro et al., 2012), which has 13,320 YouTube clips of people doing 101 different actions. We only use eight exercise classes: body-weight squats, lunges, push-ups, wall push-ups, pull-ups, handstand push-ups, jumping jacks and jump rope.

UCF101 groups its clips by the original video they came from, so we split the data by group. This way the same person never shows up in both training and testing.

| Split      | Clips | Groups |
|------------|-------|--------|
| Train      | 552   | 8–21 |
| Validation | 151   | 22–25 |
| Test       | 263   | 1–7 |
| Total      | 966   |        |

Problem Statement:
When you work out at home, there's no one there to tell you what exercise you're doing, count your reps or correct your form, and most fitness apps make you stop and type things in. We want to fix that by training a model that recognizes the exercise from video on its own. Our goal is at least 88% accuracy with predictions coming in under 0.5 seconds.

Some challenges we expect:
- Some exercises look alike. A squat and a lunge can look almost the same in one frame, and you only see the difference in how the person moves.
- The dataset is small (about 140 clips per exercise), so the model could overfit.
- If clips from the same video end up in both training and testing, the results would look better than they really are.
- We are training on laptops without GPUs, so we keep the pretrained part of the model frozen to save time.
- The YouTube videos have good lighting and cameras, but our demo will use a webcam in a classroom, so live accuracy will probably be lower.


Steps to Completion:

    Preprocessing – done once and shared with the whole team.
        Use OpenCV to pull 32 evenly spaced frames from each clip and resize them to 224×224.
        Run every frame through ResNet50, a CNN that was already trained on ImageNet. It turns each frame into a list of 2048 numbers that describe what's in the picture.
        Save these features so nobody has to process the videos again when running experiments.

    Base Model
        A CNN-only model as a reference, to show why we need something that understands movement over time.
        Our main model, a CNN-LSTM: the LSTM (256 units) reads the features from 16 frames in order and learns how the movement changes. After that comes dropout and a final layer that picks one of the eight exercises. We train it with the Adam optimizer (learning rate 0.001) and stop early if the validation loss stops improving.

    Experiments – each team member changes one thing from the base model and checks how it affects accuracy and speed.

| Experiment       | What we change                                        | Why                                           |
|------------------|-------------------------------------------------------|-----------------------------------------------|
| Backbone         | ResNet50 vs. MobileNetV2                              | A lighter CNN might be fast enough for live use |
| Frames per clip  | 8 vs. 16 vs. 32                                       | More frames show more movement but are slower |
| Input resolution | 112 vs. 224 px                                        | Bigger images take the most computing power |
| Temporal model   | LSTM vs. GRU vs. bidirectional LSTM; 128 / 256 / 512 units | Bigger models can learn more but might overfit |
| Regularization   | Dropout rate and where it goes                        | To help with overfitting |
| Optimizer        | Adam vs. SGD; learning rate                           | How fast and how well the model learns |
| Soft voting      | Averaging predictions over a few windows of the same clip | More stable predictions (Bang & Park, 2024) |

    Choosing the Final Model
        We pick the version with the best accuracy that can still predict in under 0.5 seconds.
        A change only counts as an improvement if accuracy or macro F1 goes up and the model is still fast enough.

    Live Demo
        A Streamlit web app that runs the model on a webcam, shows the exercise it sees, keeps a workout log, and says the exercise and count out loud.
        Stretch goal: count reps and give form tips using MediaPipe pose tracking (for example, checking knee angle to see how deep a squat is).


Evaluation Metrics:
We test every model on the same test set.
    Accuracy – how many test clips the model gets right. Goal: 88% or higher.
    Macro F1-score – balances precision and recall across all eight exercises equally.
    Confusion matrix – shows which exercises the model mixes up.
    Inference speed – how long one prediction takes. Goal: 0.5 seconds or less.
We show the results for each experiment in before-and-after tables.


Deliverables:
    Proposal (2%) – Due Oct 26, 2026
    Mid-Progress Meeting (8%) – Due Nov 16, 2026
        What we've done so far, experiments and results, problems we ran into, updated timeline, early results with graphs (accuracy/loss curves, confusion matrix), and feedback from the instructor.
    Final Report and Presentation (24%) – Due Dec 14, 2026
        The report covers the problem, the model, what we optimized and why, results compared to the base model, challenges, conclusion and future work.
        The presentation covers the same things with visuals, followed by Q&A.
        One person submits for the whole group.


Timeline:

| Week | Dates          | What's happening |
|------|----------------|------------------|
| 1    | Oct 19–25      | Finalize the plan; research; download UCF101 and make the splits; extract frames and features; set up the repo |
| 2    | Oct 26–Nov 1   | Submit proposal (Oct 26); train the base CNN-LSTM and CNN-only model (Oct 29); start experiments (Oct 29) |
| 3    | Nov 2–8        | Everyone pushes at least one experiment result (Nov 8) |
| 4    | Nov 9–15       | Mid-progress slides and figures (Nov 14); finished experiment notebooks pushed and combined (Nov 15) |
| 5    | Nov 16–22      | Mid-progress meeting (Nov 16); make changes based on feedback; pick the final model |
| 6    | Nov 23–29      | Streamlit demo (Nov 24); first report draft (Nov 26); plan the presentation and assign parts (Nov 28) |
| 7    | Nov 30–Dec 6   | Final report draft (Nov 30); finish the web app (Dec 2); build the final presentation (Dec 6) |
| 8    | Dec 7–13       | Final scripts (Dec 10); 2-hour practice session (Dec 11–13, TBD) |
| —    | Dec 14         | Submit final report and presentation |


References:
Bang, G.-S., & Park, S.-B. (2024). Workout classification using a convolutional neural network in ensemble learning. Sensors, 24(10), 3133. https://doi.org/10.3390/s24103133
Donahue, J., Hendricks, L. A., Guadarrama, S., Rohrbach, M., Venugopalan, S., Saenko, K., & Darrell, T. (2015). Long-term recurrent convolutional networks for visual recognition and description. CVPR, 2625–2634. https://doi.org/10.1109/CVPR.2015.7298878
He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. CVPR, 770–778. https://doi.org/10.1109/CVPR.2016.90
Paul, S. (2021, May 28). Video classification with a CNN-RNN architecture. Keras. https://keras.io/examples/vision/video_classification/
Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L.-C. (2018). MobileNetV2: Inverted residuals and linear bottlenecks. CVPR, 4510–4520. https://doi.org/10.1109/CVPR.2018.00474
Soomro, K., Zamir, A. R., & Shah, M. (2012). UCF101: A dataset of 101 human actions classes from videos in the wild (arXiv:1212.0402). arXiv. https://doi.org/10.48550/arXiv.1212.0402