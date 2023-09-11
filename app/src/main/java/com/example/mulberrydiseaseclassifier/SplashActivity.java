package com.example.mulberrydiseaseclassifier;

import android.content.Intent;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.view.View;
import android.view.animation.Animation;
import android.view.animation.AnimationUtils;
import android.widget.ImageView;
import android.widget.ProgressBar;

import androidx.appcompat.app.AppCompatActivity;


public class SplashActivity extends AppCompatActivity {

    ImageView imageView, imageView2;
    ProgressBar progressBar;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_splash);

        imageView = findViewById(R.id.splash_logo);
        imageView2 = findViewById(R.id.splash_circle);

        progressBar = findViewById(R.id.splash_progressBar);



        Animation rotateAnimation = AnimationUtils.loadAnimation(SplashActivity.this, R.anim.accelerate_rotate);
        Animation rotateAnimation2 = AnimationUtils.loadAnimation(SplashActivity.this, R.anim.accelerate_rotate);
        imageView.startAnimation(rotateAnimation);
        imageView2.startAnimation(rotateAnimation2);

        Handler handler =new Handler(Looper.getMainLooper());
        handler.postDelayed(new Runnable() {
            @Override
            public void run() {

                progressBar.setVisibility(View.GONE);
                Intent intent =new Intent(SplashActivity.this, com.example.mulberrydiseaseclassifier.MainActivity.class);
                startActivity(intent);
                finish();

            }
        },2000);


    }

    @Override
    protected void onStart() {
        super.onStart();

    }

    @Override
    protected void onResume() {
        super.onResume();


    }

    @Override
    protected void onPause() {
        super.onPause();


    }


    @Override
    protected void onDestroy() {
        super.onDestroy();

        finish();
    }



}