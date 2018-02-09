library(dplyr)
library(ggplot2)
require(gridExtra)
library(scales)

gg_color_hue <- function(n) {
  hues = seq(15, 375, length = n + 1)
  hcl(h = hues, l = 65, c = 100)[1:n]
}

num_edits <- matrix(c(92252, 2,
                      81820, 3,
                      71973, 4,
                      61978, 5,
                      53085, 6,
                      46860, 7,
                      41210, 8,
                      36629, 9,
                      32810, 10,
                      29907, 11,
                      26947, 12,
                      24941, 13,
                      22800, 14,
                      20834, 15,
                      19000, 16,
                      17817, 17,
                      16551, 18,
                      15305, 19,
                      14217, 20,
                      13486, 21), byrow = TRUE, ncol = 2)

num_edits <- as.data.frame(num_edits) %>% 
             transmute(count = V1, edits = V2)

last_modified <- matrix(c(19018, '2017-10-30',
                          15495, '2017-10-29',
                          11640, '2017-10-26',
                          11083, '2017-10-28',
                          9994, '2017-10-27',
                          9727, '2017-10-25',
                          9142, '2017-10-24',
                          8588, '2017-10-23',
                          7953, '2017-10-22',
                          6980, '2017-10-19',
                          6407, '2017-10-21',
                          5986, '2017-10-18',
                          5979, '2017-10-20',
                          5792, '2017-10-17',
                          5653, '2017-10-16',
                          5375, '2017-10-15',
                          4840, '2017-10-12',
                          4483, '2017-10-14',
                          4460, '2017-10-11'), byrow = TRUE, ncol = 2)

last_modified <- as.data.frame(last_modified) %>% 
                 transmute(count = as.integer(V1), date = as.Date(V2))

length <- matrix(c(15057, 20,                                                                                           
                   14177, 15,                                                                                           
                   13876, 21,                                                                                           
                   13856, 16,                                                                                           
                   13685, 17,
                   13629, 18,
                   13602, 22,
                   13531, 19,
                   13250, 24,
                   13149, 23,
                   13077, 30,
                   13043, 14,
                   13031, 25,
                   12834, 26,
                   12513, 28,
                   12502, 27,
                   12332, 29,
                   12318, 13,
                   12016, 12,
                   11882, 31), byrow = TRUE, ncol = 2)

length <- as.data.frame(length) %>% 
          transmute(count = V1, length = V2)

num_followers <- matrix(c(754219, 1,
                          149600, 2,
                          46939, 3,
                          19591, 4,
                          9813, 5,
                          5360, 6,
                          3305, 7,
                          2143, 8,
                          1512, 9,
                          1006, 10,
                          825, 11,
                          632, 12,
                          479, 13,
                          359, 14,
                          328, 15,
                          290, 16,
                          235, 17,
                          207, 18,
                          162, 19,
                          138, 20), byrow = TRUE, ncol = 2)

num_followers <- as.data.frame(num_followers) %>% 
  transmute(count = V1, followers = V2)


colors <- gg_color_hue(4)
plen <- ggplot(length, aes(x = length, y = count)) + 
  geom_col(fill = colors[1]) + 
  ggtitle("Length of Playlist")
plm <- ggplot(last_modified, aes(x = date, y = count)) + 
  geom_col(fill = colors[2]) +
  ggtitle("Last Modified") +
  scale_x_date(breaks = date_breaks("weeks"))
pne <- ggplot(num_edits, aes(x = edits, y = count)) + 
  geom_col(fill = colors[3]) + 
  ggtitle("Number of edits")
pnf <- ggplot(num_followers, aes(x = followers, y = count)) + 
  geom_col(fill = colors[4]) + 
  ggtitle("Number of followers")

grid.arrange(plen, plm, pne, pnf, nrow = 2)

