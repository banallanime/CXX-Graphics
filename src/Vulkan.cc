#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <vector>

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <vulkan/vulkan.hpp>

#define VK_CHECK(func)                                            \
	if (auto err = func) {                                        \
		std::cout << vk::to_string(vk::Result(err)) << std::endl; \
		return err;                                               \
	}
#define VK_LOG(message) std::cout << message << std::endl
// Boiler plate functions
VkPhysicalDevice GetPhysicalDevice(const std::vector<VkPhysicalDevice> pAdapters);
uint32_t GetBufferMemoryType(const VkDevice device, const VkBuffer buffer, VkMemoryPropertyFlags flags);

VkShaderModule CompileShader(const char* file_name, const VkDevice device);
// Constants
const char* APPLICATION_NAME = "CXX-Graphics Vulkan";
constexpr int WINDOW_WIDTH = 480, WINDOW_HEIGHT = 480;

int main(void)
{
	glfwInit();
	glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
	GLFWwindow* window = glfwCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "Vulkan-Model", nullptr, nullptr);

	if (window == nullptr) {
		return -1;
	}
	// START HERE

	// The instance represents an instance of a platform's vulkan implementation.
	static VkInstance vk_instance = nullptr;
	// The device represents an instance of a device's vulkan driver.
	static VkPhysicalDevice vk_adapter = nullptr;
	static VkDevice vk_device = nullptr;
	static VkQueue vk_queue = nullptr;
	static uint32_t selected_queue = 0;
	// 1) Create instance
	{
		// Every vulkan structure requires a VK_STRUCTURE_TYPE value.
		// Well, it doesn't require it. But the validation layers will
		// constantly warn you.

		// VkApplicationInfo is optional, and mostly exist to help drivers
		// apply patches to your application.
		VkApplicationInfo app_info {
			.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
			.pNext = nullptr,
			.pApplicationName = APPLICATION_NAME,
			.applicationVersion = VK_MAKE_API_VERSION(0, 1, 0, 0),
			.pEngineName = nullptr,
			.engineVersion = 0,
			// This specifies the maximum vulkan API version. This field is required.
			.apiVersion = VK_MAKE_API_VERSION(0, 1, 1, 0)
		};
		// Layers can change the behaviour of vulkan commands.
		// The main usage for this is the validation layer, used for debugging
		// your application and reporting any undefined behavior.
		const char* layer_names[] = { "VK_LAYER_KHRONOS_validation" };
		uint32_t ext_count = 0;
		// Extensions, well, extend vulkan.
		// Do note that there is a difference between VkInstance extensions, and
		// VkDevice extensions. Extensions only have to be enabled for one or
		// the other.
		const char** ext_names = glfwGetRequiredInstanceExtensions(&ext_count);
		VkInstanceCreateInfo instance_info {
			.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
			.pNext = nullptr,
			.flags = 0,
			.pApplicationInfo = &app_info,

			.enabledLayerCount = 1,
			.ppEnabledLayerNames = layer_names,

			.enabledExtensionCount = ext_count,
			.ppEnabledExtensionNames = ext_names,
		};

		VK_CHECK(vkCreateInstance(&instance_info, nullptr, &vk_instance));
		VK_LOG("Created VkInstance");
	}
	// 2) Create the device.
	{
		// Graphic APIS make a distinction between a physical device
		// and a virtual device. The physical device doesn't do anything.
		// It just holds information about the device and its driver.
		uint32_t adapters_count = 0;
		std::vector<VkPhysicalDevice> adapters;

		// Enumeration commands in vulkan go through a two phase process
		// 1: Get the count. This requires passing an [IN_OUT] p...Count
		// member and having the [OUT] p... member set to nullptr.
		// Make sure when passing it for getting the count, count is
		// set to 0. Or else vulkan wont be happy :(
		VK_CHECK(vkEnumeratePhysicalDevices(vk_instance, &adapters_count, nullptr));

		// Also, you can pass less than the maximum enumeration count.
		// BUT, vulkan will return VK_INCOMPLETE warning.
		adapters.resize(adapters_count);
		VK_CHECK(vkEnumeratePhysicalDevices(vk_instance, &adapters_count, adapters.data()));

		// Choose the physical device
		vk_adapter = GetPhysicalDevice(adapters);

		if (vk_adapter == nullptr) {
			std::cout << "Can't find VkPhysicalDevice" << std::endl;
			return -1;
		}

		// Vulkan also has the concepts of queue families.
		// These basically describe different parts of the device hardware,
		// and what they are capable of. An example of when this may be
		// useful is on Nvidia GPUs where they have special transfer queues.
		// This can be used in games streaming assets from another I/O thread.

		// But every vulkan driver should at least support 1 graphics queue
		// that can do mostly everything. Lets find that queue.
		uint32_t queue_count = 0;
		vkGetPhysicalDeviceQueueFamilyProperties(
			vk_adapter,
			&queue_count,
			nullptr);
		auto queue_properties = std::vector<VkQueueFamilyProperties>(queue_count);
		vkGetPhysicalDeviceQueueFamilyProperties(
			vk_adapter,
			&queue_count,
			queue_properties.data());
		std::optional<uint32_t> graphics_queue_offset;
		for (uint32_t i = 0; i < queue_count; i++) {
			const auto& prop = queue_properties[i];
			if (prop.queueFlags & VK_QUEUE_GRAPHICS_BIT) {
				graphics_queue_offset = i;
				break;
			}
		}
		if (not graphics_queue_offset.has_value()) {
			std::cout << "Cannot find Graphics queue" << std::endl;
			return -1;
		}

		float useless_priority = 1.0f;
		VkDeviceQueueCreateInfo queue_info {
			.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
			.pNext = nullptr,
			.flags = 0,
			.queueFamilyIndex = graphics_queue_offset.value(),
			.queueCount = 1,
			.pQueuePriorities = &useless_priority
		};

		// We also need some device extensions
		const char* ext_names[] = {
			// If you want to display anything to the screen, you need this.
			VK_KHR_SWAPCHAIN_EXTENSION_NAME,
			// This isn't required. But trust me, you don't want to deal with render passes.
			// But if you have to <TODO: document vulkan renderpasses>
		};
		constexpr uint32_t ext_count = 1;
		// To check if a physical device actually supports an extension,
		// enumerate through vkEnumerateDeviceExtensionProperties
		// Reference:
		// https://registry.khronos.org/vulkan/specs/latest/man/html/vkEnumerateDeviceExtensionProperties.html

		// For simplicity, we will assume a lot of these features are supported.
		VkDeviceCreateInfo device_info {
			.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
			.pNext = nullptr,
			.flags = 0,
			.queueCreateInfoCount = 1,
			.pQueueCreateInfos = &queue_info,
			// Don't use layers on devices, they are deprecated
			.enabledLayerCount = 0,
			.ppEnabledLayerNames = nullptr,
			.enabledExtensionCount = ext_count,
			.ppEnabledExtensionNames = ext_names,
			// Enabled features, unlike extensions, are features
			// apart of the Vulkan spec that aren't required.
			// Does it make sense? Idk.
			.pEnabledFeatures = nullptr
		};

		VK_CHECK(vkCreateDevice(vk_adapter, &device_info, nullptr, &vk_device));
		selected_queue = graphics_queue_offset.value();
		vkGetDeviceQueue(vk_device, selected_queue, 0, &vk_queue);
		VK_LOG("Created VkDevice");
	}

	// Now we have to create our command buffers.
	// Without these, we legit cannot do anything with the device.
	// In older graphic APIs like OpenGL, these don't exist, and
	// all device commands are stored in some internal queue to
	// be handled by the drivers. Vulkan command buffers allow
	// applications to reduce a lot of cpu overhead and thread
	// synchronization. One use case is recording UI work and
	// only re-recording a command buffer when the window size
	// changes.

	constexpr uint32_t FRAME_BUFFER_COUNT = 2;
	static VkCommandBuffer vk_cmd_buffers[FRAME_BUFFER_COUNT];
	// Command pools contain and control the allocated memory of
	// command buffers. If you plan to multithread, you need one
	// command pool for that thread.
	static VkCommandPool vk_cmd_pool;
	{
		VkCommandPoolCreateInfo pool_info {
			.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
			.pNext = nullptr,
			.flags =
				// I don't know why this isn't default behavior
				// This allows you to reset command buffers instead of
				// reallocating them every time you want to use it.
			VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
			.queueFamilyIndex = selected_queue
		};

		VK_CHECK(vkCreateCommandPool(vk_device, &pool_info, nullptr, &vk_cmd_pool));
		VkCommandBufferAllocateInfo buffer_info {
			.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
			.pNext = nullptr,
			.commandPool = vk_cmd_pool,
			.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
			// A common technique in graphics programming is using 1
			// command buffer per frame. This way, you can record one
			// buffer while another command buffer is being executed.
			.commandBufferCount = FRAME_BUFFER_COUNT
		};
		vkAllocateCommandBuffers(vk_device, &buffer_info, vk_cmd_buffers);
		VK_LOG("Created command buffers");
	}

	// We want to display stuff.
	// So we need to provide information on our display
	// To make things simple, we wont be handling window resizing
	static VkSurfaceKHR vk_surface = nullptr;
	static VkSwapchainKHR vk_swapchain = nullptr;
	static VkImage vk_swapchain_images[FRAME_BUFFER_COUNT];
	static VkImageView vk_swapchain_image_views[FRAME_BUFFER_COUNT];
	// We need to do thread synchronization so that we arent
	// trying to record an active command buffer.
	static VkFence vk_fences[FRAME_BUFFER_COUNT];
	// We also need to make sure on the GPU side the fences dont
	// execute at the same time. Semaphores are basically fences
	// for the GPU.
	static VkSemaphore vk_semaphores_swapchain[FRAME_BUFFER_COUNT];
	// Because vulkan is the worst API ever, we have to have two
	// semaphores for different purposes. One for determining when
	// the swapchain image is available for rendering, and the other
	// for determining when it is finished by us for drawing.
	// If you don't use the second one, it's going to try to present
	// while the image is already being used.
	static VkSemaphore vk_semaphores_present[FRAME_BUFFER_COUNT];
	{

		VK_CHECK(glfwCreateWindowSurface(vk_instance, window, NULL, &vk_surface));
		VkSurfaceCapabilitiesKHR surface_caps {};
		VK_CHECK(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(vk_adapter, vk_surface, &surface_caps));
		std::cout << surface_caps.currentExtent.width << std::endl;
		VkSwapchainCreateInfoKHR swapchain_info {
			.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
			.pNext = nullptr,
			.flags = 0,
			.surface = vk_surface,
			.minImageCount = FRAME_BUFFER_COUNT,
			.imageFormat = VK_FORMAT_R8G8B8A8_SRGB,
			.imageColorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR,
			.imageExtent = {
				.width = 480,
				.height = 480 },
			.imageArrayLayers = 1,
			// Annoying that we even have to specify this...
			// This basically just says the image is being used for coloring I guess.
			.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
			// We dont care about these.
			.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE,
			.queueFamilyIndexCount = 0,
			.pQueueFamilyIndices = nullptr,
			// Basically describes how an image should be flipped/mirrored.
			.preTransform = surface_caps.currentTransform,
			// If you are working with a windows compisitor and want to
			// utilize it, this may be of interest.
			// reference: https://registry.khronos.org/vulkan/specs/latest/man/html/VkCompositeAlphaFlagBitsKHR.html
			.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
			// Basically how the swapchain reacts to the present command.
			// Only VK_PRESENT_MODE_FIFO_KHR is guaranteed.
			// So in real code, check for that.
			// reference: https://registry.khronos.org/vulkan/specs/latest/man/html/VkPresentModeKHR.html
			.presentMode = VK_PRESENT_MODE_MAILBOX_KHR,
			.clipped = VK_TRUE,
			// You should provide this when reconstructing the swapchain
			.oldSwapchain = nullptr
		};
		VK_CHECK(
			vkCreateSwapchainKHR(
				vk_device,
				&swapchain_info,
				nullptr,
				&vk_swapchain));
		// Next we have to get the swapchain image handles.
		auto tmp_count = FRAME_BUFFER_COUNT;
		VK_CHECK(vkGetSwapchainImagesKHR(vk_device, vk_swapchain, &tmp_count, vk_swapchain_images));
		// Thanks Vulkan FOR WASTING MY TIME. Why, just why the hell, WOULD YOU NEED 3 DIFFERENT OBJECTS
		// TO REPRESENT IMAGE MEMORY???? THIS BENEFITS NOBODY. "Oh but the mip-" I HAD TO ATTACH THAT INFORMATION
		// TO THE GOD DAMN IMAGE OBJECT. Theres explicit for optimization, THEN THERES WASTING EVERYBODYS TIME
		// FOR NO REASON. WHAT A PILE OF GARBAGE API. JUST WORTHLESS.
		//
		// Oh btw, because Vulkan is RETARDED and THE WORST GRAPHICS PROGRAMMING API TO EVER EXIST, you need
		// all three. VkImage, VkImageView, VkFrameBuffer. 🖕🖕🖕🖕🖕🖕
		VkImageViewCreateInfo views_info {
			.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
			.pNext = nullptr,
			.flags = 0,
			// WHAT WAS EVEN THE POINT. WHY DIDNT YOU JUST ADD THIS TO VK IMAGE TYPE
			// ENUMERATIONS
			.viewType = VK_IMAGE_VIEW_TYPE_2D,
			// WHY DO I HAVE TO KEEP TRACK OF THE FORMAT? WHAT WAS THE GOD DAMN POINT
			// OF ME EVEN CREATING THE IMAGE THEN?????
			.format = VK_FORMAT_R8G8B8A8_SRGB,
			// Set these to 0 (VK_COMPONENT_SWIZZLE_IDENTITY) unless you plan
			// to remap rgba components
			.components = {
				.r = VK_COMPONENT_SWIZZLE_IDENTITY,
				.g = VK_COMPONENT_SWIZZLE_IDENTITY,
				.b = VK_COMPONENT_SWIZZLE_IDENTITY,
				.a = VK_COMPONENT_SWIZZLE_IDENTITY },
			.subresourceRange = { .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .baseMipLevel = 0, .levelCount = 1, .baseArrayLayer = 0, .layerCount = 1 }
		};

		VkFenceCreateInfo fence_info {
			.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
			.pNext = nullptr,
			.flags = VK_FENCE_CREATE_SIGNALED_BIT
		};

		VkSemaphoreCreateInfo semaphore_info {
			.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
			.pNext = nullptr,
			.flags = 0
		};

		for (size_t i = 0; i < FRAME_BUFFER_COUNT; i++) {
			views_info.image = vk_swapchain_images[i];
			VK_CHECK(vkCreateImageView(vk_device, &views_info, nullptr, &vk_swapchain_image_views[i]));
			VK_CHECK(vkCreateFence(vk_device, &fence_info, nullptr, &vk_fences[i]));
			VK_CHECK(vkCreateSemaphore(vk_device, &semaphore_info, nullptr, &vk_semaphores_swapchain[i]));
			VK_CHECK(vkCreateSemaphore(vk_device, &semaphore_info, nullptr, &vk_semaphores_present[i]));
		}
		VK_LOG("Created VkSwapchainKHR");
	} // END
	// This is so boring man
	// Pipeline shaders
	VkShaderModule vert_shader = nullptr;
	VkShaderModule frag_shader = nullptr;
	// If you don't know what a graphics pipeline or shaders are:
	// Shader: https://www.khronos.org/opengl/wiki/Shader
	// Pipeline: https://www.khronos.org/opengl/wiki/Rendering_Pipeline_Overview
	// TLDR: Shaders are GPU programs, and pipeline uses these shaders
	// to transform the input data.
	static VkPipeline vk_3d_pipeline = nullptr;
	{
		vert_shader = CompileShader(
			"resources/shaders/3d_basic.frag.spv",
			vk_device);
		frag_shader = CompileShader(
			"resources/shaders/3d_basic.frag.spv",
			vk_device);
		VkPipelineShaderStageCreateInfo vert_stage {
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.pNext = nullptr,
			.flags = 0,
			.stage = VK_SHADER_STAGE_VERTEX_BIT,
			.module = vert_shader,
		};
		VkGraphicsPipelineCreateInfo pipeline_info {
			.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
			.pNext = nullptr,
			.flags = 0,
			.stageCount = 2,
		};
		VK_LOG("Compiled shaders");
	}

	struct VertexData {
		glm::vec3 position;
		glm::vec3 color;
	};
	// Create the vertex and index buffers for our cube
	// We dont need VkBufferViews
	// (but we need image views and framebuffers for rendering... right 🖕)
	VkBuffer vk_cube_vert_buffer = nullptr;
	VkBuffer vk_cube_idx_buffer = nullptr;
	VkBuffer vk_cube_tform_buffer = nullptr;
	// To avoid unneeded memory allocations, lets put all our data in
	// one allocation block.
	VkDeviceMemory vk_cube_memory;
	{
		const std::vector<VertexData> cube_vertices = {
			// Front face
			{ { -1, -1, 1 }, { 1, 0, 0 } }, // Red
			{ { 1, -1, 1 }, { 1, 1, 0 } }, // Red+Green
			{ { 1, 1, 1 }, { 1, 1, 1 } }, // White (RGB)
			{ { -1, 1, 1 }, { 1, 0, 1 } }, // Red+Blue

			// Back face
			{ { -1, -1, -1 }, { 0, 0, 0 } }, // Black
			{ { 1, -1, -1 }, { 0, 1, 0 } }, // Green
			{ { 1, 1, -1 }, { 0, 1, 1 } }, // Green+Blue
			{ { -1, 1, -1 }, { 0, 0, 1 } }, // Blue
		};

		// Indices to form triangles
		const std::vector<uint32_t> rgbCubeIndices = {
			// Front
			0, 1, 2, 2, 3, 0,
			// Right
			1, 5, 6, 6, 2, 1,
			// Back
			5, 4, 7, 7, 6, 5,
			// Left
			4, 0, 3, 3, 7, 4,
			// Top
			3, 2, 6, 6, 7, 3,
			// Bottom
			4, 5, 1, 1, 0, 4
		};
		VkBufferCreateInfo buffer_info {
			.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
			.pNext = nullptr,
			.flags = 0,
			.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
			.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
			.queueFamilyIndexCount = 0,
			.pQueueFamilyIndices = nullptr
		};
		buffer_info.size = cube_vertices.size();
		// A device >>can<< expose different memory types for specific types
		// of resources. An example is Nvidia, whos architectures has different
		// types for general memory, and texture memory.
		// reference:
		// - https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#memory-hierarchy
		// - https://www.nvidia.com/en-us/technologies/
		// Another example is SAM/ ReBAR memory.
		// TODO: copy paste the explanation from VulkanX
		// reference:
		VK_LOG("Created cube model");
	}

	// We need to set up some MORE horrible boiler plate.
	// And they all have to be named data because c++ removes
	// compound literals :DDD
	const VkCommandBufferBeginInfo cmd_begin_info {
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
		.pNext = nullptr,
		.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
		.pInheritanceInfo = nullptr
	};
	// Every frame we want to set the framebuffer to a color.
	// This way, we remove all data from the previous frame
	const VkClearColorValue clear_color = {
		1.0f, 0.0f, 0.0f, 0.0f
	};
	VkPresentInfoKHR present_info {
		.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
		.pNext = nullptr,
		.waitSemaphoreCount = 1,
		.swapchainCount = 1,
		.pSwapchains = &vk_swapchain,
	};
	
	// Finally, we can start doing loop stuff
	uint32_t current_frame = 0;
	uint32_t current_image = 0;
	present_info.pImageIndices = &current_image;

	// LOOP
	while (not glfwWindowShouldClose(window)) {
		
		uint32_t previous_frame = current_frame;
		current_frame = (++current_frame) % FRAME_BUFFER_COUNT;
		auto& current_fence = vk_fences[current_frame];
		// This semaphore is for waiting for the swapchain image
		// to be available for drawing
		auto& image_available = vk_semaphores_swapchain[current_frame];
		// and this one is for waiting for it to be available for
		// presenting
		auto& image_available_present = vk_semaphores_present[current_frame];
		
		// Wait for the previous frame to finish if it's still active
		vkWaitForFences(vk_device, 1, &current_fence, VK_TRUE, -1);
		vkResetFences(vk_device, 1, &current_fence);
		
		vkAcquireNextImageKHR(
			vk_device,
			vk_swapchain,
			UINT64_MAX, image_available, current_fence, &current_image);
		// Wait for the acquire next image to finish.
		// If you don't have a fence waiting for this command, high
		// chance you will get validation and rendering
		vkWaitForFences(vk_device, 1, &current_fence, VK_TRUE, -1);
		vkResetFences(vk_device, 1, &current_fence);
		
		auto& command_buffer = vk_cmd_buffers[current_image];
		vkResetCommandBuffer(command_buffer, 0);

		vkBeginCommandBuffer(command_buffer, &cmd_begin_info);
		// vkCmdBindVertexBuffers(command_buffer, 0, 1, &vk_cube_vert_buffer, nullptr);
		// vkCmdBindIndexBuffer(command_buffer, vk_cube_idx_buffer, 0, VK_INDEX_TYPE_UINT32);
		VkImageSubresourceRange subresourceRange = {
			VK_IMAGE_ASPECT_COLOR_BIT,
			0, 1, 0, 1
		};

		VkImageMemoryBarrier toGeneralBarrier = {
			VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, // sType
			nullptr, // pNext
			0, // srcAccessMask
			VK_ACCESS_TRANSFER_WRITE_BIT, // dstAccessMask
			VK_IMAGE_LAYOUT_UNDEFINED, // oldLayout
			VK_IMAGE_LAYOUT_GENERAL, // newLayout
			VK_QUEUE_FAMILY_IGNORED, // srcQueueFamilyIndex
			VK_QUEUE_FAMILY_IGNORED, // dstQueueFamilyIndex
			vk_swapchain_images[current_image], // image
			subresourceRange // subresourceRange
		};

		vkCmdPipelineBarrier(
			command_buffer,
			VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
			VK_PIPELINE_STAGE_TRANSFER_BIT,
			0,
			0, nullptr,
			0, nullptr,
			1, &toGeneralBarrier);

		vkCmdClearColorImage(
			command_buffer,
			vk_swapchain_images[current_image],
			VK_IMAGE_LAYOUT_GENERAL,
			&clear_color,
			1,
			&subresourceRange);

		VkImageMemoryBarrier toPresentBarrier = {
			VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
			nullptr,
			VK_ACCESS_TRANSFER_WRITE_BIT,
			0,
			VK_IMAGE_LAYOUT_GENERAL,
			VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
			VK_QUEUE_FAMILY_IGNORED,
			VK_QUEUE_FAMILY_IGNORED,
			vk_swapchain_images[current_image],
			subresourceRange
		};

		vkCmdPipelineBarrier(
			command_buffer,
			VK_PIPELINE_STAGE_TRANSFER_BIT,
			VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
			0,
			0, nullptr,
			0, nullptr,
			1, &toPresentBarrier);

		vkEndCommandBuffer(command_buffer);
		// Submission info
		// This tells Vulkan that the semaphor is signaled when all
		// commands are finished
		const VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;

		VkSubmitInfo submit_info {
			.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
			.pNext = nullptr,
			.waitSemaphoreCount = 1,
			.pWaitSemaphores = &image_available,
			.pWaitDstStageMask = &wait_stage,
			.commandBufferCount = 1,
			.pCommandBuffers = &command_buffer,
			.signalSemaphoreCount = 1,
			.pSignalSemaphores = &image_available_present
		};

		vkQueueSubmit(vk_queue, 1, &submit_info, current_fence);
		present_info.pImageIndices = &current_image;
		present_info.pWaitSemaphores = &image_available_present;
		vkQueuePresentKHR(vk_queue, &present_info);		
		glfwPollEvents();
	}
	vkDeviceWaitIdle(vk_device);
	// You can skip this
	// Destroy all vulkan objects
	for (uint32_t i = 0; i < FRAME_BUFFER_COUNT; i++) {
		
		vkDestroySemaphore(vk_device, vk_semaphores_swapchain[i],nullptr);
		vkDestroySemaphore(vk_device, vk_semaphores_present[i],nullptr);
		vkDestroyFence(vk_device, vk_fences[i], nullptr);
		vkDestroyImageView(vk_device, vk_swapchain_image_views[i], nullptr);
	}
	
	vkDestroySwapchainKHR(vk_device, vk_swapchain, nullptr);
	vkDestroySurfaceKHR(vk_instance, vk_surface, nullptr);
	glfwDestroyWindow(window);
	vkDestroyCommandPool(vk_device, vk_cmd_pool, nullptr);
	vkDestroyDevice(vk_device, nullptr);
	vkDestroyInstance(vk_instance, nullptr);
	glfwTerminate();
	/*
	Congrats! You learnt the basics of Vulkan.
	Here are some things you might want to do in a Vulkan renderer:

	- Decide the minimum Vulkan version required
	(for android, or if you're targetting older devices,
	you want this to be pretty low).
	- Choosing the right memory type for textures and buffers
	(or just use VMA: https://gpuopen-librariesandsdks.github.io/VulkanMemoryAllocator/html/index.html)
	- Support Rebar/Sam/UMA allocators
	(or just use VMA)
	- Properly detect supported surface format(s), and os specific
	display format.
	- Nvidia/AMD specific optimizations
	(Example, Nvidia's transfer queue)
	- Learn the mess that is render passes
	- Ray tracing support
	(reference: https://developer.nvidia.com/rtx/raytracing/vkray)
	- Mesh shader support

	As a closing note: FUCK YOU VULKAN 🖕🖕🖕🖕🖕
	*/
	return 0;
}

// Boiler plate functions impl
using std::vector;
VkPhysicalDevice GetPhysicalDevice(const vector<VkPhysicalDevice> pAdapters)
{
	std::optional<uint32_t> integrated_offset, discrete_offset;

	// Choose the first discrete, else integrated
	for (int i = 0; i < pAdapters.size(); i++) {
		const auto& adapter = pAdapters[i];
		VkPhysicalDeviceProperties properties;
		vkGetPhysicalDeviceProperties(adapter, &properties);
		if (properties.deviceType & VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
			discrete_offset = i;
			break;
		}
		if (properties.deviceType & VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU) {
			integrated_offset = i;
		}
	}
	uint32_t offset = 0;
	if (discrete_offset.has_value()) {
		offset = discrete_offset.value();
	} else if (integrated_offset.has_value()) {
		offset = integrated_offset.value();
	} else {
		return nullptr;
	}
	return pAdapters[offset];
};

VkShaderModule CompileShader(const char* file_name, const VkDevice device)
{
	if (not std::filesystem::exists(file_name)) {
		std::cout << "Invalid file: " << file_name << std::endl;
		return nullptr;
	};
	// Pointer needs to be u32 aligned
	uint32_t* data = nullptr;
	size_t fsize = std::filesystem::file_size(file_name);
	// Fix file size to be properly aligned
	const size_t fsize_fixed = (fsize / 2) + (fsize % 2);
	data = new uint32_t[fsize_fixed];
	try {
		std::ifstream file(file_name);
		file.read(reinterpret_cast<char*>(data), fsize);
		file.close();
	} catch (...) {
		std::cout << "Cannot open file: " << file_name << std::endl;
		return nullptr;
	}

	VkShaderModuleCreateInfo shader_info {
		.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
		.pNext = nullptr,
		.flags = 0,
		.codeSize = fsize,
		.pCode = data
	};
	VkShaderModule shader_module;
	// if (vkCreateShaderModule(device, &shader_info, nullptr, &shader_module) < VK_SUCCESS) {
	// 	return nullptr;
	// }
	return nullptr;
}

uint32_t GetBufferMemoryType(const VkDevice device, const VkBuffer buffer)
{

	return 0;
};
