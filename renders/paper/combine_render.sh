GAP=10
W=128
H=128

LEFT_PAD=150
TOP_PAD=40

FONT="/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

# --------------------------
# Falsecolor version
# --------------------------
ffmpeg \
  -i hsi_mipnerf_MD_GR_20_0_128x48fov_Falsecolor.mp4 \
  -i hsi_mipnerf_MD_GR_40_0_128x48fov_Falsecolor.mp4 \
  -i hsi_mipnerf_MD_GR_50_4_128x48fov_Falsecolor.mp4 \
  -i hsi_mipnerf_MD_GR_100_3_128x48fov_Falsecolor.mp4 \
  -i hsi_mipnerf_L2_20_3_128x48fov_Falsecolor.mp4 \
  -i hsi_mipnerf_L2_40_4_128x48fov_Falsecolor.mp4 \
  -i hsi_mipnerf_L2_50_4_128x48fov_Falsecolor.mp4 \
  -i hsi_mipnerf_L2_100_0_128x48fov_Falsecolor.mp4 \
  -filter_complex "\
    xstack=inputs=8:layout=\
0_0|\
$((W+GAP))_0|\
$(((W+GAP)*2))_0|\
$(((W+GAP)*3))_0|\
0_$((H+GAP))|\
$((W+GAP))_$((H+GAP))|\
$(((W+GAP)*2))_$((H+GAP))|\
$(((W+GAP)*3))_$((H+GAP)):fill=black[grid]; \
    [grid]pad=iw+$LEFT_PAD:ih+$TOP_PAD:$LEFT_PAD:$TOP_PAD:black[p0]; \
    [p0]drawtext=fontfile=$FONT:text='20 imgs':x=$((LEFT_PAD + 0*(W+GAP) + 10)):y=20:fontsize=14:fontcolor=white[p1]; \
    [p1]drawtext=fontfile=$FONT:text='40 imgs':x=$((LEFT_PAD + 1*(W+GAP) + 10)):y=20:fontsize=14:fontcolor=white[p2]; \
    [p2]drawtext=fontfile=$FONT:text='50 imgs':x=$((LEFT_PAD + 2*(W+GAP) + 10)):y=20:fontsize=14:fontcolor=white[p3]; \
    [p3]drawtext=fontfile=$FONT:text='100 imgs':x=$((LEFT_PAD + 3*(W+GAP) + 10)):y=20:fontsize=14:fontcolor=white[p4]; \
    [p4]drawtext=fontfile=$FONT:text='Our Method':x=20:y=$((TOP_PAD + H/2 - 18)):fontsize=18:fontcolor=white[p5]; \
    [p5]drawtext=fontfile=$FONT:text='Mip-NeRF':x=20:y=$((TOP_PAD + H + GAP + H/2 - 18)):fontsize=18:fontcolor=white[outv] \
  " \
  -map "[outv]" \
#   -c:v libopenh264 -b:v 30M -maxrate 30M -bufsize 60M -g 1 -keyint_min 1 -pix_fmt yuv420p \
    -c:v libopenh264 -b:v 30M -maxrate 30M -bufsize 60M -g 15 -keyint_min 15 -r 30 -vsync cfr -pix_fmt yuv420p \
  grid_Falsecolor.mp4

# --------------------------
# ACE version (only filenames/output name changed)
# --------------------------
ffmpeg \
  -i hsi_mipnerf_MD_GR_20_0_128x48fov_ACE.mp4 \
  -i hsi_mipnerf_MD_GR_40_0_128x48fov_ACE.mp4 \
  -i hsi_mipnerf_MD_GR_50_4_128x48fov_ACE.mp4 \
  -i hsi_mipnerf_MD_GR_100_3_128x48fov_ACE.mp4 \
  -i hsi_mipnerf_L2_20_3_128x48fov_ACE.mp4 \
  -i hsi_mipnerf_L2_40_4_128x48fov_ACE.mp4 \
  -i hsi_mipnerf_L2_50_4_128x48fov_ACE.mp4 \
  -i hsi_mipnerf_L2_100_0_128x48fov_ACE.mp4 \
  -filter_complex "\
    xstack=inputs=8:layout=\
0_0|\
$((W+GAP))_0|\
$(((W+GAP)*2))_0|\
$(((W+GAP)*3))_0|\
0_$((H+GAP))|\
$((W+GAP))_$((H+GAP))|\
$(((W+GAP)*2))_$((H+GAP))|\
$(((W+GAP)*3))_$((H+GAP)):fill=black[grid]; \
    [grid]pad=iw+$LEFT_PAD:ih+$TOP_PAD:$LEFT_PAD:$TOP_PAD:black[p0]; \
    [p0]drawtext=fontfile=$FONT:text='20 imgs':x=$((LEFT_PAD + 0*(W+GAP) + 10)):y=20:fontsize=14:fontcolor=white[p1]; \
    [p1]drawtext=fontfile=$FONT:text='40 imgs':x=$((LEFT_PAD + 1*(W+GAP) + 10)):y=20:fontsize=14:fontcolor=white[p2]; \
    [p2]drawtext=fontfile=$FONT:text='50 imgs':x=$((LEFT_PAD + 2*(W+GAP) + 10)):y=20:fontsize=14:fontcolor=white[p3]; \
    [p3]drawtext=fontfile=$FONT:text='100 imgs':x=$((LEFT_PAD + 3*(W+GAP) + 10)):y=20:fontsize=14:fontcolor=white[p4]; \
    [p4]drawtext=fontfile=$FONT:text='Our Method':x=20:y=$((TOP_PAD + H/2 - 18)):fontsize=18:fontcolor=white[p5]; \
    [p5]drawtext=fontfile=$FONT:text='Mip-NeRF':x=20:y=$((TOP_PAD + H + GAP + H/2 - 18)):fontsize=18:fontcolor=white[outv] \
  " \
  -map "[outv]" \
  -c:v libopenh264 -b:v 30M -maxrate 30M -bufsize 60M -g 1 -keyint_min 1 -pix_fmt yuv420p \
  grid_ACE.mp4
