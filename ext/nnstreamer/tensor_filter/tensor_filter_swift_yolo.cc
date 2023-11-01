/**
 * GStreamer Tensor_Filter, swift_yolo Module
 *
 */
/**
 * @file    tensor_filter_swift_yolo.cc
 * @date    31 Oct 2023
 * @brief   nncn module for tensor_filter gstreamer plugin
 * @see     http://github.com/nnstreamer/nnstreamer
 * @author  Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug     No known bugs except for NYI items
 *
 * This is the per-NN-framework plugin (swift_yolo) for tensor_filter.
 *
 */

#include <nnstreamer_log.h>
#include <nnstreamer_plugin_api_util.h>
#define NO_ANONYMOUS_NESTED_STRUCT
#include <nnstreamer_plugin_api_filter.h>
#undef NO_ANONYMOUS_NESTED_STRUCT

#include <nnstreamer_conf.h>
#include <nnstreamer_util.h>

#include <net.h>


/**
 * @brief Possible swift-yolo delegates.
 */
typedef enum {
  TFLITE_DELEGATE_NONE = 0,
  TFLITE_DELEGATE_GPU,
  TFLITE_DELEGATE_NNAPI,
  TFLITE_DELEGATE_XNNPACK,
  TFLITE_DELEGATE_EXTERNAL,

  TFLITE_DELEGATE_MAX
} yolos_delegate_e;

/**
 * @brief Option to open tf-lite model.
 */
typedef struct {
  const gchar *model_file[2]; /**< path to swift-yolo model file */
  const gchar *accelerators; /**< accelerators set for this subplugin */
  yolos_delegate_e delegate; /**< swift-yolo delegate */
  gint num_threads; /**< the number of threads */
  const gchar *ext_delegate_path; /**< path to external delegate lib */
  GHashTable *ext_delegate_kv_table; /**< external delegate key values options */
} yolos_option_s;

static const gchar *yolos_accl_support[] = { ACCL_CPU_STR, ACCL_GPU_STR, NULL };

/**
 * @brief Tensor Filter NNCN Module
 */
class YolosCore
{
    public:
    YolosCore (const char **_model_path);
    ~YolosCore ();

    int init (yolos_option_s *option);
    int loadModel ();
    int getInputTensorDim (GstTensorsInfo *info);
    int getOutputTensorDim (GstTensorsInfo *info);

    private:
    char **model_path;
    int num_threads;
    
    GstTensorsInfo inputTensorMeta; /**< The tensor info of input tensors */
    GstTensorsInfo outputTensorMeta; /**< The tensor info of output tensors */
}

extern "C" { /* accessed by android api */
void init_filter_yolos (void) __attribute__ ((constructor));
void fini_filter_yolos (void) __attribute__ ((destructor));
}

YolosCore::YolosCore (const char **_model_path)
{
  g_assert (_model_path != NULL);
  model_path = g_strdup (_model_path);

  gst_tensors_info_init (&inputTensorMeta);
  gst_tensors_info_init (&outputTensorMeta);
}

/**
 * @brief	initialize the object with tflite model
 * @param	option options to initialize tf-lite model
 * @return 0 if OK. non-zero if error.
 *        -1 if the model is not loaded.
 *        -2 if the initialization of input tensor is failed.
 *        -3 if the initialization of output tensor is failed.
 *        -4 if the caching of input and output tensors failed.
 */
int
YolosCore::init (tflite_option_s *option)
{
  num_threads = option->num_threads;
  int err;

  setAccelerator (option->accelerators, option->delegate);
  g_message ("accl = %s", get_accl_hw_str (accelerator));

  if ((err = loadModel ())) {
    ml_loge ("Failed to load model (swift-yolo interpreter->loadModel() has returned %d. Please check if the model, '%s', is accessible and compatible with the given swift-yolo instance. For example, this swift-yolo's version might not support the given model.\n",
        err, option->model_file);
    return -1;
  }
  return 0;
}

/**
 * @brief	get the model path
 * @return the model path.
 */
const char **
YolosCore::getModelPath ()
{
  return model_path;
}

/**
 * @brief	load the tflite model
 * @note	the model will be loaded
 * @return 0 if OK. non-zero if error.
 */
int
YolosCore::loadModel ()
{
  int err;
  ncnn::Net::load_param(model_path[0]);
  ncnn::Net::load_model(model_path[1]);
  return err;
}

/**
 * @brief	return the Dimension of Input Tensor.
 * @param[out] info Structure for tensor info.
 * @return 0 if OK. non-zero if error.
 */
int
YolosCore::getInputTensorDim (GstTensorsInfo *info)
{
  gst_tensors_info_copy (info, &inputTensorMeta);
  return 0;
}

/**
 * @brief	return the Dimension of Tensor.
 * @param[out] info Structure for tensor info.
 * @return 0 if OK. non-zero if error.
 */
int
YolosCore::getOutputTensorDim (GstTensorsInfo *info)
{
  gst_tensors_info_copy (info, &outputTensorMeta);
  return 0;
}

/**
 * @brief	invoke the model with the input.
 * @param[in] input : The array of input tensors
 * @param[out]  output : The array of output tensors
 * @return 0 if OK. non-zero if error.
 *         -1 if the input properties are incompatible.
 *         -2 if the output properties are different with model.
 *         -3 if the output is neither a list nor a tensor.
 *         -4 if running the model failed.
 */
int
YolosCore::invoke (const GstTensorFilterProperties *prop,
    const GstTensorMemory *input, GstTensorMemory *output)
{
#if (DBG)
  gint64 start_time = g_get_real_time ();
#endif
  GstTensorInfo *_info;
  ncnn::Mat in_pad;
  ncnn::Mat out;
  ncnn::Extractor ex = ncnn::Net::create_extractor();
  at::Tensor tensor;

  /** @todo Support other input types other than at::Tensor */
  for (uint i = 0; i < inputTensorMeta.num_tensors; ++i) {
    std::vector<int64_t> input_shape;

    _info = gst_tensors_info_get_nth_info (&inputTensorMeta, i);

    input_shape.assign (&_info->dimension[0], &_info->dimension[0] + NNS_TENSOR_RANK_LIMIT);

    if (!getTensorTypeToTorch (_info->type, &type)) {
      ml_loge ("This data type is not valid: %d", _info->type);
      return -1;
    }
    input_shape.resize (prop->input_ranks[i]);
    std::reverse (input_shape.begin (), input_shape.end ());
    in_pad = ncnn::Mat::from_pixels_resize(input[i].data, ncnn::Mat::PIXEL_RGB2BGR, input_shape[0], input_shape[1], input_shape[0], input_shape[1]);
    ex.input("images", in_pad);
    ex.extract("output", out);
    memcpy (output[i].data, out, output[i].size);
  }
  return 0;
}

/**
 * @brief Free privateData and move on.
 */
static void
yolos_close (const GstTensorFilterProperties *prop, void **private_data)
{
  YolosCore *core = static_cast<YolosCore *> (*private_data);
  UNUSED (prop);

  if (!core)
    return;

  delete core;

  *private_data = NULL;
}

/**
 * @brief Load swift_yolo modelfile
 * @param prop property of tensor_filter instance
 * @param private_data : swift_yolo plugin's private data
 * @return 0 if successfully loaded. 1 if skipped (already loaded).
 *        -1 if the object construction is failed.
 *        -2 if the object initialization if failed
 */
static gint
yolos_loadModelFile (const GstTensorFilterProperties *prop, void **private_data)
{
  YolosCore *core;
  const gchar *model_path[2];

  if (prop->num_models != 2) {
    ret = -1;
    goto done;
  }

  if (yolos_parseCustomOption (prop, &option) != 0) {
    g_printerr ("Failed to parse options to initialize swift-yolo model.");
    ret = -1;
    goto done;
  }
  core = static_cast<YolosCore *> (*private_data);
  model_path = prop->model_files;

  if (core != NULL) {
    if (g_strcmp0 (model_path, core->getModelPath ()) == 0) {
      ret = 1; /* skipped */
      goto done;
    }

    yolos_close (prop, private_data);
  }

  core = new YolosCore (model_path);
  if (core == NULL) {
    g_printerr ("Failed to allocate memory for filter subplugin: swift_yolo\n");
    ret = -1;
    goto done;
  }

  if (core->init (&option) != 0) {
    *private_data = NULL;
    delete core;

    g_printerr ("failed to initialize the object: swift-yolo");
    ret = -2;
    goto done;
  }

  *private_data = core;

done:
  g_free ((gpointer) option.ext_delegate_path);
  option.ext_delegate_path = nullptr;

  if (option.ext_delegate_kv_table)
    g_hash_table_unref (option.ext_delegate_kv_table);
  option.ext_delegate_kv_table = nullptr;

  return ret;
}

/**
 * @brief The open callback for GstTensorFilterFramework. Called before anything else
 * @param prop property of tensor_filter instance
 * @param private_data : swift_yolo plugin's private data
 */
static gint
yolos_open (const GstTensorFilterProperties *prop, void **private_data)
{
  gint status = yolos_loadModelFile (prop, private_data);

  return status;
}

/**
 * @brief The mandatory callback for GstTensorFilterFramework
 * @param prop property of tensor_filter instance
 * @param private_data : swift_yolo plugin's private data
 * @param[in] input The array of input tensors
 * @param[out] output The array of output tensors
 * @return 0 if OK. non-zero if error.
 */
static gint
yolos_invoke (const GstTensorFilterProperties *prop, void **private_data,
    const GstTensorMemory *input, GstTensorMemory *output)
{
  YolosCore *core = static_cast<YolosCore *> (*private_data);
  g_return_val_if_fail (core && input && output, -EINVAL);

  return core->invoke (prop, input, output);
}

/**
 * @brief The optional callback for GstTensorFilterFramework
 * @param prop property of tensor_filter instance
 * @param private_data : swift_yolo plugin's private data
 * @param[out] info The dimesions and types of input tensors
 */
static gint
yolos_getInputDim (const GstTensorFilterProperties *prop, void **private_data,
    GstTensorsInfo *info)
{
  YolosCore *core = static_cast<YolosCore *> (*private_data);
  g_return_val_if_fail (core && info, -EINVAL);
  UNUSED (prop);

  return core->getInputTensorDim (info);
}

/**
 * @brief The optional callback for GstTensorFilterFramework
 * @param prop property of tensor_filter instance
 * @param private_data : swift_yolo plugin's private data
 * @param[out] info The dimesions and types of output tensors
 */
static gint
yolos_getOutputDim (const GstTensorFilterProperties *prop, void **private_data,
    GstTensorsInfo *info)
{
  YolosCore *core = static_cast<YolosCore *> (*private_data);
  g_return_val_if_fail (core && info, -EINVAL);
  UNUSED (prop);

  return core->getOutputTensorDim (info);
}

/**
 * @brief The optional callback for GstTensorFilterFramework
 * @param[in] hw backend accelerator hardware
 * @return 0 if supported. -errno if not supported.
 */
static int
yolos_checkAvailability (accl_hw hw)
{
  if (g_strv_contains (yolos_accl_support, get_accl_hw_str (hw)))
    return 0;

  return -ENOENT;
}

static gchar filter_subplugin_swift_yolo[] = "swift_yolo";

static GstTensorFilterFramework NNS_support_yolos = { .version = GST_TENSOR_FILTER_FRAMEWORK_V0,
  .open = yolos_open,
  .close = yolos_close,
  { .v0 = {
        .name = filter_subplugin_swift_yolo,
        .allow_in_place = FALSE, /** @todo: support this to optimize performance later. */
        .allocate_in_invoke = FALSE,
        .run_without_model = FALSE,
        .verify_model_path = TRUE, /* check that the given .pt files are valid */
        .statistics = nullptr,
        .invoke_NN = yolos_invoke,
        .getInputDimension = yolos_getInputDim,
        .getOutputDimension = yolos_getOutputDim,
        .setInputDimension = nullptr,
        .destroyNotify = nullptr,
        .reloadModel = nullptr,
        .handleEvent = nullptr,
        .checkAvailability = yolos_checkAvailability,
        .allocateInInvoke = nullptr,
    } } };

/** @brief Initialize this object for tensor_filter subplugin runtime register */
void
init_filter_yolos (void)
{
  nnstreamer_filter_probe (&NNS_support_swift_yolo);
  // todo: add custom property
}

/** @brief Destruct the subplugin */
void
fini_filter_yolos (void)
{
  nnstreamer_filter_exit (NNS_support_swift_yolo.v0.name);
}