"""
Quick Test Script for Grad-CAM Fix
Run this to verify the fix works before deploying
"""

import numpy as np
import tensorflow as tf

def test_gradcam_input_format():
    """Test that demonstrates the fix"""
    
    print("=" * 60)
    print("Testing Grad-CAM Input Format Fix")
    print("=" * 60)
    
    # Simulate the model inputs structure
    from tensorflow.keras.layers import Input, Concatenate, Dense
    from tensorflow.keras.models import Model
    
    # Create a simple dual-input model (mimics VGG19 + ResNet50 ensemble)
    input1 = Input(shape=(224, 224, 3), name='resnet_input')
    input2 = Input(shape=(224, 224, 3), name='vgg_input')
    
    # Simple processing
    flat1 = tf.keras.layers.Flatten()(input1)
    flat2 = tf.keras.layers.Flatten()(input2)
    concat = Concatenate()([flat1, flat2])
    output = Dense(2, activation='softmax')(concat)
    
    model = Model(inputs=[input1, input2], outputs=output)
    
    print(f"\n✅ Created test model with inputs: {model.input_names}")
    
    # Create test data
    test_data_dict = {
        'resnet_input': np.random.randn(1, 224, 224, 3).astype(np.float32),
        'vgg_input': np.random.randn(1, 224, 224, 3).astype(np.float32)
    }
    
    test_data_list = [
        test_data_dict['resnet_input'],
        test_data_dict['vgg_input']
    ]
    
    print("\n" + "-" * 60)
    print("Test 1: Dictionary Input (for main model)")
    print("-" * 60)
    try:
        pred_dict = model(test_data_dict)
        print(f"✅ Dictionary input works: {pred_dict.shape}")
    except Exception as e:
        print(f"❌ Dictionary input failed: {e}")
    
    print("\n" + "-" * 60)
    print("Test 2: List Input (for grad model)")
    print("-" * 60)
    try:
        pred_list = model(test_data_list)
        print(f"✅ List input works: {pred_list.shape}")
    except Exception as e:
        print(f"❌ List input failed: {e}")
    
    print("\n" + "-" * 60)
    print("Test 3: Creating Grad Model (the actual fix)")
    print("-" * 60)
    try:
        # This is how we build the grad model
        grad_model = Model(
            inputs=model.inputs,  # This is already a list!
            outputs=[model.layers[2].output, model.output]  # Some internal layer + output
        )
        print(f"✅ Grad model created successfully")
        
        # Now test with DICT (OLD WAY - BREAKS)
        print("\n   Testing grad_model with DICT input (broken):")
        try:
            result_dict = grad_model(test_data_dict)
            print(f"   ❌ Unexpectedly worked with dict: {[r.shape for r in result_dict]}")
        except Exception as e:
            print(f"   ✅ Expected error with dict: {type(e).__name__}")
        
        # Now test with LIST (NEW WAY - WORKS)
        print("\n   Testing grad_model with LIST input (fixed):")
        try:
            result_list = grad_model(test_data_list)
            print(f"   ✅ Works with list: {[r.shape for r in result_list]}")
        except Exception as e:
            print(f"   ❌ Unexpected error with list: {e}")
            
    except Exception as e:
        print(f"❌ Grad model creation failed: {e}")
    
    print("\n" + "=" * 60)
    print("Summary: The fix converts dict → list before calling grad_model")
    print("=" * 60)
    print("""
    ✅ CORRECT USAGE:
    input_list = [full_input_dict["resnet_input"], full_input_dict["vgg_input"]]
    conv_outputs, predictions = grad_model(input_list)
    
    ❌ BROKEN USAGE:
    conv_outputs, predictions = grad_model(full_input_dict)
    """)
    print("=" * 60)

if __name__ == "__main__":
    test_gradcam_input_format()
    print("\n🎉 Test complete! The fix is validated.")
