from scipy import misc


im1 = misc.imread("images/joey1.jpg")
im2 = misc.imread("images/joey2.jpg")

im1 = misc.imresize(im1, (96, 96))
im2 = misc.imresize(im2, (96, 96))

misc.imsave('images/joey1_reshape.jpeg', im1)
misc.imsave('images/joey2_reshape.jpeg', im2)

