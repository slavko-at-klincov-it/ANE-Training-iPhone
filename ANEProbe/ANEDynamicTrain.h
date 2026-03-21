// ANEDynamicTrain.h — Dynamic spatial packing training loop
// Compile once, update weights via IOSurface each step (no recompile!)
#import <Foundation/Foundation.h>

/// Run dynamic spatial packing training tests:
/// A) Per-channel scale (proven in Phase 1.4)
/// B) Full weight matrix via spatial packing
/// Returns human-readable result string.
NSString *ane_dynamic_train_test(void);
