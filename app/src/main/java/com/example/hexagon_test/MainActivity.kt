package com.example.hexagon_test

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.tooling.preview.Preview
import com.example.hexagon_test.ui.theme.Hexagon_testTheme

class MainActivity : ComponentActivity() {
    companion object {
        init {
//            System.loadLibrary("OpenCL")
//            System.loadLibrary("OpenCL_system")
//            System.loadLibrary("cutils")
//            System.loadLibrary("vndksupport")
            System.loadLibrary("hexagon")
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            Hexagon_testTheme {
                // A surface container using the 'background' color from the theme
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colorScheme.background
                ) {
                    Greeting("Android")
                }
            }
        }
    }
}

@Composable
fun Greeting(name: String, modifier: Modifier = Modifier) {
    val hexagon = Hexagon()

    Text(
        text = hexagon.checkRuntime(),
        modifier = modifier
    )
}

@Preview(showBackground = true)
@Composable
fun GreetingPreview() {
    Hexagon_testTheme {
        Greeting("Android")
    }
}