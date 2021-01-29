#include <dlib/cmd_line_parser.h>
#include <dlib/data_io.h>
#include <dlib/dnn.h>

// Whenever two rectangles in boxes overlap, according to overlaps(), we set the
// smallest box to ignore and return the number of newly ignored boxes.
auto ignore_overlapped_boxes(
    std::vector<dlib::mmod_rect>& boxes,
    const dlib::test_box_overlap& overlaps) -> int;

// make sure boxes don't spill out of the image
auto force_boxes_inside_images(
    const std::vector<dlib::matrix<dlib::rgb_pixel>>& images,
    std::vector<std::vector<dlib::mmod_rect>>& boxes) -> void;

// clean up dataset (fix boxes positions, remove overlapped and small boxes)
auto clean_dataset(
    const std::vector<dlib::matrix<dlib::rgb_pixel>>& images,
    std::vector<std::vector<dlib::mmod_rect>>& boxes) -> void;

// definition of the neural network model used for detection
namespace detector
{
    using namespace dlib;
    // clang-format off
    // ACT must be an activation layer and BN a batch normalization
    template <template <typename> class ACT, template <typename> class BN>
    struct def
    {
        // a padded convolution with custom kernel size and stride
        template <long nf, int ks, int s, typename SUBNET>
        using pcon = add_layer<con_<nf, ks, ks, s, s, ks/2, ks/2>, SUBNET>;
        // convolutional block: convolution + batch_norm + activation
        template <long num_filters, int ks, int s, typename SUBNET>
        using conblock = ACT<BN<pcon<num_filters, ks, s, SUBNET>>>;
        // this block downsamples the input by a factor of 8
        template <typename SUBNET>
        using downsampler = conblock<16, 3, 2, conblock<16, 5, 2, conblock<16, 7, 2, SUBNET>>>;
        // this block extracts more features
        template <typename SUBNET>
        using extractor = conblock<32, 3, 1, conblock<32, 3, 1, conblock<32, 3, 1, SUBNET>>>;
        // the network type using the Max-Margin Object Detector loss
        using net_type = loss_mmod<con<1, 9, 9, 1, 1, extractor<downsampler<input_rgb_image>>>>;
    };
    // the training version of the network with ReLU and Batch Normalization
    using train = def<relu, bn_con>::net_type;
    // the inference version of the network with ReLU and Affine Layer
    using infer = def<relu, affine>::net_type;
    // clang-format on
}  // namespace detector

int main(const int argc, const char** argv)
try
{
    dlib::command_line_parser parser;
    parser.add_option("batch-size", "mini batch size for training (default: 32)", 1);
    parser.add_option("data", "path to data with training.xml file", 1);
    parser.add_option("gpus", "number of GPUs to train on (default: 1)", 1);
    parser.set_group_name("Help Options");
    parser.add_option("h", "alias for --help");
    parser.add_option("help", "display this message and exit");
    parser.parse(argc, argv);

    if (parser.option("h") or parser.option("help"))
    {
        std::cout << "Usage: " << argv[0] << " OPTIONS..." << std::endl;
        parser.print_options();
        return EXIT_SUCCESS;
    }

    const size_t mini_batch_size = dlib::get_option(parser, "batch-size", 32);
    const size_t num_gpus = dlib::get_option(parser, "gpus", 1);
    const std::string data_dir = dlib::get_option(parser, "data", "");
    if (data_dir.empty())
    {
        std::cout << "Provide the data directory with --data" << std::endl;
        return EXIT_FAILURE;
    }

    // load dataset
    std::vector<dlib::matrix<dlib::rgb_pixel>> images_train;
    std::vector<std::vector<dlib::mmod_rect>> boxes_train;
    dlib::load_image_dataset(images_train, boxes_train, data_dir + "/training.xml");
    dlib::add_image_left_right_flips(images_train, boxes_train);
    std::clog << "Loaded " << images_train.size() << " images" << std::endl;

    // adjust boxes size
    clean_dataset(images_train, boxes_train);

    // set up the options for the MMOD loss at a single scale (no image pyramid)
    // this will find clusters of anchor boxes, and we consider a match if IOU > 0.55
    dlib::mmod_options options(dlib::use_image_pyramid::no, boxes_train, 0.55);
    // ignore bounding boxes whose IOU > 0.5 or are 95% covered by another bbox
    options.overlaps_ignore = dlib::test_box_overlap(0.5, 0.95);
    // use bounding box regression to have boxes better fitted to detected objects
    options.use_bounding_box_regression = true;
    // instatiation of the detector network in train mode
    detector::train net(options);
    // remove duplicatives biases: notably biases from convultional layers followed by batch norm
    dlib::disable_duplicative_biases(net);
    // set up the number of filters in the last layer (num classes + 4 for bounding box regression)
    net.subnet().layer_details().set_num_filters(options.detector_windows.size() * 5);
    // log the network to see the detailed definition
    std::cout << net << std::endl;

    std::vector<int> gpus(num_gpus);
    std::iota(gpus.begin(), gpus.end(), 0);
    // define a trainer for this network (by default it uses SGD with 0.0005 wd and 0.9 momentum)
    auto trainer = dlib::dnn_trainer(net, dlib::sgd(), gpus);
    trainer.set_learning_rate(0.1);
    trainer.be_verbose();
    trainer.set_mini_batch_size(mini_batch_size);
    trainer.set_iterations_without_progress_threshold(5000);
    trainer.set_synchronization_file("football_detector_sync", std::chrono::minutes(30));
    std::cout << trainer << std::endl;

    // start the training with minibatches
    decltype(images_train) mini_batch_samples;
    decltype(boxes_train) mini_batch_labels;
    dlib::rand rnd;
    while (trainer.get_learning_rate() >= 1e-4)
    {
        // make sure all images in the minibatch have the same size
        const auto generate_mini_batch = [&]() {
            mini_batch_samples.clear();
            mini_batch_labels.clear();
            size_t i = 0;
            while (i < trainer.get_mini_batch_size())
            {
                int index = rnd.get_random_32bit_number() % images_train.size();
                if (mini_batch_samples.empty() or
                    images_train[index].size() == mini_batch_samples[0].size())
                {
                    mini_batch_samples.push_back(images_train[index]);
                    mini_batch_labels.push_back(boxes_train[index]);
                    ++i;
                }
            }
        };
        generate_mini_batch();
        // some data augmentation
        for (auto&& img : mini_batch_samples)
            dlib::disturb_colors(img, rnd);

        trainer.train_one_step(mini_batch_samples, mini_batch_labels);
    }
    // save the network to disk
    trainer.get_net();
    net.clean();
    dlib::serialize("football_detector.dnn") << net;
    // load the network, but in inference (test) mode
    detector::infer tnet;
    dlib::deserialize("football_detector.dnn") >> tnet;
    std::cout << dlib::test_object_detection_function(
        tnet,
        images_train,
        boxes_train,
        dlib::test_box_overlap(),
        0, // detection score threshold
        options.overlaps_ignore);
}
catch (const std::exception& e)
{
    std::cout << e.what() << std::endl;
    return EXIT_FAILURE;
}

auto ignore_overlapped_boxes(
    std::vector<dlib::mmod_rect>& boxes,
    const dlib::test_box_overlap& overlaps) -> int
{
    int num_ignored = 0;
    for (size_t i = 0; i < boxes.size(); ++i)
    {
        if (boxes[i].ignore)
            continue;
        for (size_t j = i + 1; j < boxes.size(); ++j)
        {
            if (boxes[j].ignore)
                continue;
            if (overlaps(boxes[i], boxes[j]))
            {
                ++num_ignored;
                if (boxes[i].rect.area() < boxes[j].rect.area())
                    boxes[i].ignore = true;
                else
                    boxes[j].ignore = true;
            }
        }
    }
    return num_ignored;
}

auto force_boxes_inside_images(
    const std::vector<dlib::matrix<dlib::rgb_pixel>>& images,
    std::vector<std::vector<dlib::mmod_rect>>& boxes) -> void
{
    DLIB_CASSERT(images.size() == boxes.size());
    for (size_t i = 0; i < images.size(); ++i)
    {
        const long image_width = images[i].nc();
        const long image_height = images[i].nr();
        for (auto& box : boxes[i])
        {
            box.rect.set_left(std::max(0l, box.rect.left()));
            box.rect.set_top(std::max(0l, box.rect.top()));
            box.rect.set_right(std::min(image_width - 1, box.rect.right()));
            box.rect.set_bottom(std::min(image_height - 1, box.rect.bottom()));
        }
    }
}

auto clean_dataset(
    const std::vector<dlib::matrix<dlib::rgb_pixel>>& images,
    std::vector<std::vector<dlib::mmod_rect>>& boxes) -> void
{
    force_boxes_inside_images(images, boxes);
    size_t num_overlapped_ignored = 0;
    size_t num_small_size_ignored = 0;
    for (auto& v : boxes)
    {
        num_overlapped_ignored += ignore_overlapped_boxes(v, dlib::test_box_overlap(0.5, 0.99));
        for (auto& bb : v)
        {
            if (bb.rect.width() < 24 and bb.rect.height() < 24)
            {
                if (not bb.ignore)
                {
                    bb.ignore = true;
                    ++num_small_size_ignored;
                }
            }
        }
    }
    std::clog << "num_overlapped_ignored: " << num_overlapped_ignored << std::endl;
    std::clog << "num_small_size_ignored: " << num_small_size_ignored << std::endl;
}
