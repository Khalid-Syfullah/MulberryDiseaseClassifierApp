package com.example.mulberrydiseaseclassifier;

import static java.lang.Math.min;

import android.Manifest;
import android.app.Activity;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.RectF;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.SystemClock;
import android.os.Trace;
import android.provider.MediaStore;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.annotation.RequiresApi;
import androidx.appcompat.app.AppCompatActivity;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.gpu.CompatibilityList;
import org.tensorflow.lite.gpu.GpuDelegate;
import org.tensorflow.lite.support.common.FileUtil;
import org.tensorflow.lite.support.common.TensorOperator;
import org.tensorflow.lite.support.common.TensorProcessor;
import org.tensorflow.lite.support.common.ops.NormalizeOp;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;
import org.tensorflow.lite.support.image.ops.ResizeWithCropOrPadOp;
import org.tensorflow.lite.support.label.TensorLabel;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.ByteArrayOutputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;

public class MulberryScannerClassifierActivity extends Activity {
    private static final String TAG = "MulberryScannerActivity";
    private static final int MY_CAMERA_PERMISSION_CODE = 100;
    private static final int CAMERA_REQUEST = 1888;

    private TensorImage inputTensorImage;
    private  int imageSizeX;
    private  int imageSizeY;
    private TensorBuffer probabilityImageBuffer;
    private TensorProcessor probabilityProcessor;
    private static final float IMAGE_MEAN = 0.0f;
    private static final float IMAGE_STD = 255.0f;
    private static final float PROBABILITY_MEAN = 0.0f;
    private static final float PROBABILITY_STD = 1.0f;
    private Bitmap bitmap;
    private List<String> labels;
    ImageView imageView;
    Uri imageuri;
    Button classifierBtn;
    TextView classText, accuracyText, timeText, selectImageText;
    private ImageView backBtn;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_mulberry_disease_classifier);

        imageView=(ImageView)findViewById(R.id.plant_upload_image);
        classifierBtn =(Button)findViewById(R.id.plant_check_button);
        classText =(TextView)findViewById(R.id.plant_classify_text);
        accuracyText=(TextView)findViewById(R.id.plant_classify_text_2);
        timeText=(TextView)findViewById(R.id.plant_classify_text_3);
        selectImageText=(TextView)findViewById(R.id.plant_select_image_text);

        imageView.setImageResource(R.drawable.camera);

        backBtn=findViewById(R.id.plant_back_button);


        classText.setText("Press 'CHECK' button to classify");
        accuracyText.setText("Accuracy range: 0.0 - 1.0");
        timeText.setText("Time unit is displayed in ms");

        classText.setVisibility(View.VISIBLE);
        accuracyText.setVisibility(View.VISIBLE);
        timeText.setVisibility(View.VISIBLE);
        selectImageText.setVisibility(View.VISIBLE);


        byte[] byteArray = getIntent().getByteArrayExtra("image");
        if(byteArray != null) {
            Bitmap bmp = BitmapFactory.decodeByteArray(byteArray, 0, byteArray.length);
            imageView.setImageBitmap(bmp);
            bitmap = bmp;
            selectImageText.setText("Image is ready for classification!");

        }
        else{
            Toast.makeText(this,"Failed to load image!", Toast.LENGTH_LONG).show();
            selectImageText.setVisibility(View.GONE);


        }
        classifierBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {

                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                    imageClassifierWithSupportLibrary();
                }

            }
        });



        backBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                MulberryScannerClassifierActivity.super.onBackPressed();
            }
        });

        imageView.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {

                classText.setTextColor(getResources().getColor(R.color.blue));
                accuracyText.setTextColor(getResources().getColor(R.color.blue));
                timeText.setTextColor(getResources().getColor(R.color.blue));
                classifierBtn.setBackground(getResources().getDrawable(R.drawable.button_style_red));


                classText.setText("Press 'CHECK' button to classify");
                accuracyText.setText("Accuracy range: 0.0 - 1.0");
                timeText.setText("Time unit is displayed in ms");
                classifierBtn.setText("Check");

                classText.setVisibility(View.VISIBLE);
                accuracyText.setVisibility(View.VISIBLE);
                timeText.setVisibility(View.VISIBLE);
                selectImageText.setVisibility(View.GONE);

                if (checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED)
                {
                    requestPermissions(new String[]{Manifest.permission.CAMERA}, MY_CAMERA_PERMISSION_CODE);
                }
                else
                {
                    Intent cameraIntent = new Intent(android.provider.MediaStore.ACTION_IMAGE_CAPTURE);
                    startActivityForResult(cameraIntent, CAMERA_REQUEST);
                }
            }
        });


    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults)
    {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == MY_CAMERA_PERMISSION_CODE)
        {
            if (grantResults[0] == PackageManager.PERMISSION_GRANTED)
            {
                Toast.makeText(this, "Camera permission granted!", Toast.LENGTH_LONG).show();
                Intent cameraIntent = new Intent(android.provider.MediaStore.ACTION_IMAGE_CAPTURE);
                startActivityForResult(cameraIntent, CAMERA_REQUEST);
            }
            else
            {
                Toast.makeText(this, "Camera permission denied!", Toast.LENGTH_LONG).show();
            }
        }
    }


    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data)
    {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode == CAMERA_REQUEST && resultCode == Activity.RESULT_OK)
        {
            Toast.makeText(this,"Scan successful!", Toast.LENGTH_LONG).show();
            Bitmap photo = (Bitmap) data.getExtras().get("data");
            imageView.setImageBitmap(photo);
            selectImageText.setVisibility(View.VISIBLE);

        }
        else{
            Toast.makeText(this,"Scan unsuccessful!", Toast.LENGTH_LONG).show();
            imageView.setImageResource(R.drawable.camera);

        }
    }




    @RequiresApi(api = Build.VERSION_CODES.O)
    private void imageClassifierWithSupportLibrary(){
        try{

//            AssetFileDescriptor fileDescriptor = XRayActivity.this.getAssets().openFd("cxr17.tflite");
//            FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
//            FileChannel fileChannel = inputStream.getChannel();
//            long startoffset = fileDescriptor.getStartOffset();
//            long declaredLength=fileDescriptor.getDeclaredLength();
//            MappedByteBuffer tfliteModel = fileChannel.map(FileChannel.MapMode.READ_ONLY, startoffset, declaredLength);


            Interpreter.Options tfliteOptions = new Interpreter.Options();

            CompatibilityList compatList = new CompatibilityList();
            if(compatList.isDelegateSupportedOnThisDevice()){
                // if the device has a supported GPU, add the GPU delegate
                GpuDelegate.Options delegateOptions = compatList.getBestOptionsForThisDevice();
                GpuDelegate gpuDelegate = new GpuDelegate(delegateOptions);
                tfliteOptions.addDelegate(gpuDelegate);
                Log.d(TAG, "GPU supported. GPU delegate created and added to options");
            } else {
                tfliteOptions.setUseXNNPACK(true);
                Log.d(TAG, "GPU not supported. Default to CPU.");
            }


            //Loads model from the model file.
            MappedByteBuffer tfliteModel = FileUtil.loadMappedFile(MulberryScannerClassifierActivity.this,"FinalMobilenetFold1.tflite");
            Interpreter tflite = new Interpreter(tfliteModel, tfliteOptions);

            // Loads labels out from the label file.
            labels = FileUtil.loadLabels(MulberryScannerClassifierActivity.this,"labels.txt");


            // Reads type and shape of input and output tensors, respectively.
            int firstTensorImageIndex = 0;

            int[] inputImageShape = tflite.getInputTensor(firstTensorImageIndex).shape(); // {1, height, width, 3}
            DataType inputImageDataType = tflite.getInputTensor(firstTensorImageIndex).dataType();

            int[] probabilityImageShape = tflite.getOutputTensor(firstTensorImageIndex).shape(); // {1, NUM_CLASSES}
            DataType probabilityImageDataType = tflite.getOutputTensor(firstTensorImageIndex).dataType();

            imageSizeY = inputImageShape[1];
            imageSizeX = inputImageShape[2];



            // Creates the input tensor.
            TensorImage inputTensorImage = new TensorImage(inputImageDataType);
            // Loads bitmap into a TensorImage.
            inputTensorImage.load(bitmap);

            for(int i=0;i<10;i++){
                Log.d(TAG, "Without Normalization: "+inputTensorImage.getTensorBuffer().getFloatValue(i));
            }

            // Creates processor for the TensorImage.
            int cropSize = Math.min(bitmap.getWidth(), bitmap.getHeight());


            //Without normalization, what are the channel values
            int xw = bitmap.getWidth();
            int xh = bitmap.getHeight();

            int [] arr = new int[xw*xh];
            inputTensorImage.getBitmap().getPixels(arr, 0, xw, 0, 0, xw, xh);

            for(int i=0;i<arr.length;i++){

                int red, green, blue;

                red = (arr[i] >> 16) & 0xFF;
                green = (arr[i] >> 8) & 0xFF;
                blue = arr[i] & 0xFF;

                arr[i] = 0xFF << 24 | red << 16 | green << 8 | blue;


                if(i <= 10)
                    Log.d(TAG, "red["+i+"]: "+red + " green["+i+"]: "+green + " blue["+i+"]: "+blue);

            }

            Bitmap b2 = Bitmap.createBitmap(xw, xh, Bitmap.Config.ARGB_8888);
            b2.setPixels(arr, 0, xw, 0, 0, xw, xh);
            imageView.setImageBitmap(b2);


            ImageProcessor imageProcessor = new ImageProcessor.Builder()
                    .add(new ResizeWithCropOrPadOp(cropSize, cropSize))
                    .add(new ResizeOp(imageSizeX, imageSizeY, ResizeOp.ResizeMethod.NEAREST_NEIGHBOR))
                    .add(new NormalizeOp(IMAGE_MEAN, IMAGE_STD))
                    .build();

            inputTensorImage = imageProcessor.process(inputTensorImage);

            for(int i=0;i<10;i++){
                Log.d(TAG, "After Normalization: (Float) "+inputTensorImage.getTensorBuffer().getFloatValue(i));
                Log.d(TAG, "After Normalization: (Int) "+inputTensorImage.getTensorBuffer().getIntValue(i));

            }

            int [] bitmapArray = new int[imageSizeX*imageSizeY];
            inputTensorImage.getBitmap().getPixels(bitmapArray,0, imageSizeX, 0, 0, imageSizeX, imageSizeY);

            float [] tensorArray = inputTensorImage.getTensorBuffer().getFloatArray();
            int [] bitmapArray2 = new int[imageSizeX*imageSizeY];
            int j=0;

            for (int i=0;i<imageSizeX*imageSizeY*3;i=i+3){
//
//                int r = (int) (bitmapArray[i]*128) >> 16;
//                int G = (int) (bitmapArray[i]*128) >> 8;
//                int B = (int) bitmapArray[i]*128;

                int r = (int) tensorArray[i]*128;
                int G = (int) tensorArray[i+1]*128;
                int B = (int) tensorArray[i+2]*128;

                bitmapArray2[j] = 0xFF << 24 | r << 16 | G << 8 | B;
                j++;


                if(i <= 10)
                    Log.d(TAG, "r["+i+"]: "+r + " g["+i+"]: "+G + " b["+i+"]: "+B);

            }

//            Bitmap b3 = Bitmap.createBitmap(imageSizeX, imageSizeY, Bitmap.Config.ARGB_8888);
//            b3.setPixels(bitmapArray2, 0, imageSizeX, 0, 0, imageSizeX, imageSizeY);
//            imageView.setImageBitmap(b3);


            Log.d(TAG, "Byte Array: "+inputTensorImage.getTensorBuffer().getBuffer().array().length);
            Log.d(TAG, "Float Array: "+inputTensorImage.getTensorBuffer().getFloatArray().length);
            Log.d(TAG, "Int Array: "+inputTensorImage.getTensorBuffer().getIntArray().length);


            for(int i=0;i<10;i++){
                Log.d(TAG, "tensorArray["+i+"]: "+tensorArray[i] + " bitmapArray["+i+"]: "+bitmapArray[i] + " bitmapArray2["+i+"]: "+bitmapArray2[i]);

            }

            for(int i=0;i<10;i++){
                Log.d(TAG, "tensorArray["+i+"]: "+tensorArray[i] + " bitmapArray["+i+"]: "+bitmapArray[i] + " bitmapArray2["+i+"]: "+bitmapArray2[i]);

            }
            Log.d(TAG, "tensorArray size "+tensorArray.length + " bitmapArray size: "+bitmapArray.length + " bitmapArray2 size: "+bitmapArray2.length);




            // Creates the output tensor and its processor.
            TensorBuffer probabilityImageBuffer = TensorBuffer.createFixedSize(probabilityImageShape, probabilityImageDataType);


            // Creates the post processor for the output probability.
            TensorProcessor probabilityProcessor = new TensorProcessor.Builder()
                    .add(new NormalizeOp(PROBABILITY_MEAN, PROBABILITY_STD))
                    .build();


//
//            ByteBuffer inputBuffer = ByteBuffer.allocate(4 * inputTensorImage.getTensorBuffer().getFloatArray().length);
//
//            for (float value : inputTensorImage.getTensorBuffer().getFloatArray()){
//                inputBuffer.putFloat(value);
//            }
//
//
//            probabilityImageBuffer.getBuffer().rewind();
//
//            ByteBuffer probabilityBuffer = ByteBuffer.allocate(4 * probabilityImageBuffer.getFloatArray().length);
//
//            for (float value : probabilityImageBuffer.getFloatArray()){
//                probabilityBuffer.putFloat(value);
//            }


            // Runs the inference call.
            Trace.beginSection("runInference");
            long startTimeForReference = SystemClock.uptimeMillis();


            tflite.run(inputTensorImage.getBuffer(), probabilityImageBuffer.getBuffer());


            long endTimeForReference = SystemClock.uptimeMillis();
            Trace.endSection();
            Log.v(TAG, "Timecost to run model inference: " + (endTimeForReference - startTimeForReference) + " ms");


            for(int i=0; i< probabilityImageBuffer.getFloatArray().length;i++)
                Log.d(TAG, "Probability Image Buffer ["+i + "] :"+ String.format("%.8f", probabilityImageBuffer.getFloatValue(i)));





            // Gets the map of label and probability.
            Map<String, Float> labeledProbability = new TensorLabel(labels, probabilityProcessor.process(probabilityImageBuffer)).getMapWithFloatValue();


            float maxValueInMap = (Collections.max(labeledProbability.values()));

            for (Map.Entry<String, Float> entry : labeledProbability.entrySet()) {
                if (entry.getValue()==maxValueInMap) {

                    classText.setTextColor(getResources().getColor(R.color.green));
                    accuracyText.setTextColor(getResources().getColor(R.color.green));
                    timeText.setTextColor(getResources().getColor(R.color.green));

                    classText.setText("Class: "+entry.getKey());
                    accuracyText.setText("Accuracy: "+entry.getValue());
                    timeText.setText("Time: "+ (endTimeForReference - startTimeForReference) +" ms");

                    classText.setVisibility(View.VISIBLE);
                    timeText.setVisibility(View.VISIBLE);
                    accuracyText.setVisibility(View.VISIBLE);
                    selectImageText.setVisibility(View.GONE);

                    classifierBtn.setText(entry.getKey());
                    classifierBtn.setBackground(getResources().getDrawable(R.drawable.button_style_green));
                }
            }

//            for(int i=0; i<1000; i++){
//
//                Log.d(TAG, "Input Tensor Image ["+i+"]: "+inputTensorImage.getTensorBuffer().getFloatValue(i));
//
//            }
//
//            for(int i=0;i<1000;i++){
//                Log.d(TAG, "Output Tensor Buffer ["+i+"]: "+probabilityImageBuffer.getFloatValue(i));
//
//            }
//
//            Log.d(TAG, "Buffer Length: "+inputTensorImage.getTensorBuffer().getBuffer().array().length);
//

            Trace.endSection();



            Log.d(TAG, "Input Tensor Image Size: "+ inputTensorImage.getTensorBuffer().getFloatArray().length + " DataType: "+ inputTensorImage.getTensorBuffer().getDataType());
            Log.d(TAG, "Input Tensor Buffer Size: "+ tflite.getInputTensor(0).numBytes());
            Log.d(TAG, "Output Tensor Buffer Size: "+ tflite.getOutputTensor(0).numBytes());

            Log.d(TAG, "Bitmap Byte Count: "+ this.bitmap.getByteCount());
            Log.d(TAG, "Input Tensor Image Byte Count: "+ inputTensorImage.getTensorBuffer().getFloatArray().length);
            Log.d(TAG, "Input Image Shape: "+ inputImageShape[2] +  " "+ inputImageShape[1]);
            Log.d(TAG, "Input Image DataType: "+ inputImageDataType);
            Log.d(TAG, "Input Image Tensor[0]: "+ tflite.getInputTensor(0).asReadOnlyBuffer().getInt(0));
            Log.d(TAG, "Output Image Shape: "+ probabilityImageShape[1] +  " "+ probabilityImageShape[0]);
            Log.d(TAG, "Output Image DataType: "+ probabilityImageDataType);
            Log.d(TAG, "Output Image Tensor[0]: "+ tflite.getOutputTensor(0).asReadOnlyBuffer().getInt(0));

            Log.d("TensorBuffer","Input TensorBuffer Length: "+ inputTensorImage.getBuffer().toString().length());
            Log.d("TensorBuffer","Probability TensorBuffer Length: "+ probabilityImageBuffer.getBuffer().toString().length());






        }

        catch (Exception e) {
            e.printStackTrace();
        }



    }

    //    private void imageClassifierwithTaskLibrary() throws IOException {
//
//        // Initialization
//        ImageClassifier.ImageClassifierOptions options =
//                ImageClassifier.ImageClassifierOptions.builder()
//                        .setMaxResults(1)
//                        .build();
//
//        ImageClassifier imageClassifier = ImageClassifier.createFromFileAndOptions(
//                        XRayActivity.this, "cxr17.tflite", options);
//
//        Trace.beginSection("recognizeImage");
//
//        imageSizeX = 124;
//        imageSizeY = 124;
//
//        TensorImage inputImage = TensorImage.fromBitmap(normalizeBitmap(bitmap));
//        int width = bitmap.getWidth();
//        int height = bitmap.getHeight();
//
//        width = 124;
//        height = 124;
//
//        int cropSize = min(width, height);
//
//        ImageProcessingOptions imageOptions = ImageProcessingOptions.builder()
//                        // Set the ROI to the center of the image.
//                        .setRoi(
//                                new Rect(
//                                        /*left=*/ (width - cropSize) / 2,
//                                        /*top=*/ (height - cropSize) / 2,
//                                        /*right=*/ (width + cropSize) / 2,
//                                        /*bottom=*/ (height + cropSize) / 2))
//                        .build();
//
//        // Runs the inference call.
//        Trace.beginSection("runInference");
//        long startTimeForReference = SystemClock.uptimeMillis();
//
//        List<Classifications> results = imageClassifier.classify(inputImage, imageOptions);
//        long endTimeForReference = SystemClock.uptimeMillis();
//        Trace.endSection();
//        Log.v(TAG, "Timecost to run model inference: " + (endTimeForReference - startTimeForReference));
//
//
//        for(int i=0;i<results.size(); i++)
//            Log.d(TAG, "Run complete. Label: " + results.get(i).getCategories().get(0).getLabel() + " DisplayName: " + results.get(i).getCategories().get(0).getDisplayName() + " Score: " + results.get(i).getCategories().get(0).getScore() +" Index: " + results.get(i).getCategories().get(0).getIndex());
//
//        List<Recognition> result = getRecognitions(results);
//
//        for(int i=0;i<result.size();i++) {
//            classitext.setText("Class: "+result.get(i).getTitle());
//            accuracyText.setText("Accuracy: "+result.get(i).getConfidence() + "%");
//            accuracyText.setVisibility(View.VISIBLE);
//
//            imageView.setImageBitmap(normalizeBitmap(bitmap));
//        }
//        Trace.endSection();
//
//
//    }
//    private static List<Recognition> getRecognitions(List<Classifications> classifications) {
//
//        final ArrayList<Recognition> recognitions = new ArrayList<>();
//        // All the demo models are single head models. Get the first Classifications in the results.
//        for (Category category : classifications.get(0).getCategories()) {
//            recognitions.add(new Recognition("" + category.getLabel(), category.getLabel(), category.getScore(), null));
//        }
//        return recognitions;
//    }



    private ByteBuffer convertBitmapToByteBuffer(Bitmap bp) {
        ByteBuffer imgData = ByteBuffer.allocateDirect(Float.BYTES*imageSizeX*imageSizeY*3);
        imgData.order(ByteOrder.nativeOrder());
        Bitmap bitmap = Bitmap.createScaledBitmap(bp,imageSizeX,imageSizeY,true);
        int [] intValues = new int[imageSizeX*imageSizeY];

        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());

        // Convert the image to floating point.
        int pixel = 0;

        for (int i = 0; i < imageSizeX; ++i) {
            for (int j = 0; j < imageSizeY; ++j) {
                final int val = intValues[pixel++];

                imgData.putFloat(((val>> 16) & 0xFF) / 255.f);
                imgData.putFloat(((val>> 8) & 0xFF) / 255.f);
                imgData.putFloat((val & 0xFF) / 255.f);
            }
        }

        Log.d(TAG, "ByteBuffer Length: "+imgData.array().length);

        Bitmap bitmap1 = Bitmap.createBitmap(32, 32, Bitmap.Config.ARGB_8888);
        bitmap1.copyPixelsFromBuffer(imgData);
        imageView.setImageBitmap(bitmap1);

        return imgData;
    }

    private Bitmap normalizeBitmap(Bitmap bp) {
        ByteBuffer imgData = ByteBuffer.allocateDirect(Float.BYTES*imageSizeX*imageSizeY*3);
        imgData.order(ByteOrder.nativeOrder());
        Bitmap bitmap = Bitmap.createScaledBitmap(bp,imageSizeX,imageSizeY,true);
        int [] intValues = new int[imageSizeX*imageSizeY];
        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());

        // Convert the image to floating point.
        int pixel = 0;

        for (int i = 0; i < imageSizeX; ++i) {
            for (int j = 0; j < imageSizeY; ++j) {
                final int val = intValues[pixel++];

                imgData.putFloat(((val>> 16) & 0xFF) / 255.f);
                imgData.putFloat(((val>> 8) & 0xFF) / 255.f);
                imgData.putFloat((val & 0xFF) / 255.f);


            }
        }

//        int j = 0;
//        for(int i=0;i<intValues.length;i++){
//
//            intValues[i] = imgData.get(j);
//            j = j+3;
//            Log.d(TAG, "byteBuffer["+i+"]"+" : "+imgData.get(i));
//        }


        Log.d(TAG, "Bitmap Type: "+bitmap.getConfig().toString());
        Log.d(TAG, "Bitmap Size: "+bitmap.getByteCount());
        Log.d(TAG, "Buffer Size: "+imgData.array().length);
        Log.d(TAG, "Array Size: "+intValues.length);

        //Bitmap bp2 = Bitmap.createBitmap(imageSizeX,imageSizeY, Bitmap.Config.ARGB_8888);

        //int j=intValues.length-1;

        for(int i=0;i<intValues.length;i++){

//            Log.d(TAG, "IntValues["+i+"]"+" : "+intValues[i]);
            intValues[i] = intValues[i] / 255;
//            intValues[i] = intValues[j];
//            j--;

        }

        imgData.rewind();
        Bitmap bp2 = Bitmap.createBitmap(imageSizeX, imageSizeY, Bitmap.Config.ARGB_8888);
        bp2.setPixels(intValues,0,imageSizeX,0,0,imageSizeX,imageSizeY);

        Log.d(TAG, "Bitmap Size: "+ bitmap.getByteCount());

        return bp2;
    }

    /** Gets the top-k results. */
    private static List<Recognition> getTopKProbability(Map<String, Float> labelProb) {
        // Find the best classifications.
        PriorityQueue<Recognition> pq =
                new PriorityQueue<>(
                        3,
                        new Comparator<Recognition>() {
                            @Override
                            public int compare(Recognition lhs, Recognition rhs) {
                                // Intentionally reversed to put high confidence at the head of the queue.
                                return Float.compare(rhs.getConfidence(), lhs.getConfidence());
                            }
                        });

        for (Map.Entry<String, Float> entry : labelProb.entrySet()) {
            pq.add(new Recognition("" + entry.getKey(), entry.getKey(), entry.getValue(), null));
        }

        final ArrayList<Recognition> recognitions = new ArrayList<>();
        int recognitionsSize = min(pq.size(), 3);
        for (int i = 0; i < recognitionsSize; ++i) {
            recognitions.add(pq.poll());
        }
        return recognitions;
    }

    private TensorImage loadImage(final Bitmap bitmap) {
        // Loads bitmap into a TensorImage.
        inputTensorImage.load(bitmap);

        // Creates processor for the TensorImage.
        int cropSize = Math.min(bitmap.getWidth(), bitmap.getHeight());
        // TODO(b/143564309): Fuse ops inside ImageProcessor.
        ImageProcessor imageProcessor =
                new ImageProcessor.Builder()
                        .add(new ResizeWithCropOrPadOp(cropSize, cropSize))
                        .add(new ResizeOp(imageSizeX, imageSizeY, ResizeOp.ResizeMethod.NEAREST_NEIGHBOR))
                        .add(getPreprocessNormalizeOp())

                        .build();
        return imageProcessor.process(inputTensorImage);
    }

    private MappedByteBuffer loadmodelfile(Activity activity) throws IOException {
        AssetFileDescriptor fileDescriptor = activity.getAssets().openFd("cxr17.tflite");
        FileInputStream inputStream=new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel=inputStream.getChannel();
        long startoffset = fileDescriptor.getStartOffset();
        long declaredLength=fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY,startoffset,declaredLength);
    }

    private TensorOperator getPreprocessNormalizeOp() {
        return new NormalizeOp(IMAGE_MEAN, IMAGE_STD);
    }
    private TensorOperator getPostprocessNormalizeOp(){
        return new NormalizeOp(PROBABILITY_MEAN, PROBABILITY_STD);
    }

    private void showResult(){

        try{
            labels = FileUtil.loadLabels(this,"labels.txt");
        }catch (Exception e){
            e.printStackTrace();
        }
        Map<String, Float> labeledProbability =
                new TensorLabel(labels, probabilityProcessor.process(probabilityImageBuffer))
                        .getMapWithFloatValue();
        float maxValueInMap =(Collections.max(labeledProbability.values()));

        for (Map.Entry<String, Float> entry : labeledProbability.entrySet()) {
            if (entry.getValue()==maxValueInMap) {
                classText.setText(entry.getKey());
            }
        }
    }


}

