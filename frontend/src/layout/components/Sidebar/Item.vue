<script>
export default {
  name: 'MenuItem', // Component name
  functional: true, // Define as a functional component for stateless rendering
  props: {
    icon: {
      type: String,
      default: '' // Icon class name (optional)
    },
    title: {
      type: String,
      default: '' // Title text for the menu item
    }
  },
  render(h, context) {
    const { icon, title } = context.props

    return (
      <div class="menu-item">
        {icon && <svg-icon icon-class={icon} />} {/* Render icon if provided */}
        {title && <span>{title}</span>} {/* Render title if provided */}
      </div>
    )
  }
}
</script>

<style scoped>
.menu-item {
  display: flex; /* Arrange icon and text in a row */
  align-items: center; /* Align items vertically */
}

.menu-item span {
  margin-left: 8px; /* Add space between icon and text */
}
</style>
