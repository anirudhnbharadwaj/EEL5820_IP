# Wiener Filter Image Restoration Project

This project uses MATLAB to fix blurry images with a Wiener filter. It works with the Lena image and tries different ways to remove blur (average, motion, Gaussian). Here’s how to make it work and what the files do.

## How to Make the Code Work

1. **Get MATLAB Ready**
   - Make sure you have the Image Processing Toolbox (it’s needed for stuff like `imread` and `fspecial`).

2. **Set Up Folders**
   - Create a main folder for the project (e.g., `HW4`).
   - Inside it, make a folder:
     - `InputImages/` - Put all input images here (blurred and original).
     - Put all the `.m` scripts inside main folder (e.g., `HW4`).
   - The code will make an `Output/` folder for results when you run it.

3. **Add the Images**
   - Put these images in `InputImages/`:
     - `lena.bmp` (the original clear image)
     - `blurred_lena_av9.bmp` (average blur, 9x9)
     - `blurred_lena_av19.bmp` (average blur, 19x19)
     - `blurred_lena_m33_29.bmp` (motion blur, 33°, length 29)
     - `blurred_lena_m135_19.bmp` (motion blur, 135°, length 19)
     - `blurred_lena_g_11_3.bmp` (Gaussian blur, size 11, sigma 3)
     - `blurred_lena_g_19_7.bmp` (Gaussian blur, size 19, sigma 7)
   - Note: The code expects `.bmp` files with these exact names. If you have `.png` or different names, change the code to match.

4. **Run the Main Code**
   - Open MATLAB.
   - Run `wiener_deblur_all`.
   - It processes all images and saves results in `Output/`.

5. **Make Stitched Images (Optional for Report)**
   - After running `wiener_deblur_all`, run `stitch_images` in the same way.
   - This combines the original, blurry, and fixed images into single pictures for the report.

6. **Check Results**
   - Look in `Output/`:
     - `DeblurredImages/Direct/` - Fixed images with normal Wiener filter.
     - `DeblurredImages/Sequential/` - Fixed average blur images with two steps.
     - `DeblurredImages/Mismatched/` - Motion blur fixes with slightly wrong angles.
     - `Results/` - A table (`wiener_deblur_results.png`) and `.csv` file with PSNR and SSIM numbers, plus stitched images (if you ran `stitch_images`).

## Code Files

Here’s what each file does:

1. **`wiener_deblur_all.m`**
   - The main file you run.
   - It takes all blurry images, fixes them with the Wiener filter, and saves the results.
   - It tries different SNR values (0.1, 0.01, 0.001, 0.0001), does sequential fixes for average blur, and tests wrong angles for motion blur.
   - Makes a table with PSNR and SSIM scores.

2. **`parse_filename.m`**
   - A helper file.
   - Reads the image file names (like `blurred_lena_m33_29.bmp`) and figures out the blur type (average, motion, Gaussian) and settings (size, angle).

3. **`myWiener.m`**
   - Another helper file.
   - Does the actual Wiener filter work.
   - Loads an image, applies the filter with the right blur settings and SNR, and fixes it up with extra steps.

4. **`stitch_images.m`**
   - Optional file for the report.
   - Takes the blurry images and fixed images, then sticks them side-by-side into one picture (e.g., `avg_9.png`).
   - Saves these in `Output/Results/`.